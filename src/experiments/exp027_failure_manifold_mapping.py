"""
Experiment 027 — Failure Manifold Mapping
==========================================
Stop evaluating tasks. Start evaluating failures directly.
Maps the multidimensional structural boundary (the failure manifold) 
of audio representations using PCA dimensionality reduction and K-Means clustering.

Key Steps:
1. Synthesise a reference clean harmonic stack.
2. Apply 8 different types of random perturbations (2,000 samples total).
3. Compute representation similarity (cosine similarity) and extract 11 descriptors.
4. Project the 11-dimensional failure space to 2D via PCA.
5. Cluster the 2D projected space into 5 clusters using K-Means.
6. Profile the clusters and identify collapse regions.
7. Save visualization results/exp027_failure_manifold.png.
"""

import sys
import os
import numpy as np
import scipy.signal
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.representations.acf import compute_acf
from src.representations.cepstrum import compute_cepstrum
from src.representations.stft import compute_stft

# ---------------------------------------------------------------------------
# Reference Signal Synthesis
# ---------------------------------------------------------------------------

def synthesise_reference_frame(sr: int, win: int, f0: float = 220.0) -> np.ndarray:
    """
    Synthesise a clean harmonic stack frame (5 harmonics) of size win.
    """
    t = np.arange(win) / sr
    frame = np.zeros(win)
    for k in range(1, 6):
        frame += (1.0 / k) * np.sin(2 * np.pi * k * f0 * t)
    
    # Apply Hanning window
    env = np.ones(win)
    fade = int(0.02 * sr)
    env[:fade] = np.linspace(0.0, 1.0, fade)
    env[-fade:] = np.linspace(1.0, 0.0, fade)
    frame *= env
    
    frame /= np.max(np.abs(frame)) + 1e-9
    return frame


# ---------------------------------------------------------------------------
# Perturbations Generator
# ---------------------------------------------------------------------------

def apply_random_perturbation(frame: np.ndarray, sr: int, p_idx: int, rng: np.random.Generator) -> tuple[np.ndarray, str]:
    """
    Applies one of the 8 random perturbations.
    Returns (perturbed_frame, perturbation_label).
    """
    win = len(frame)
    perturbed = frame.copy()
    label = "Unknown"

    if p_idx == 0:
        # Additive Noise
        noise_std = rng.uniform(0.05, 0.80)
        perturbed = frame + rng.normal(0, noise_std, win)
        label = "Noise"
        
    elif p_idx == 1:
        # Lowpass Filter
        cutoff = rng.uniform(150, 1500)
        b, a = scipy.signal.butter(4, cutoff / (sr / 2.0), btype='low')
        perturbed = scipy.signal.filtfilt(b, a, frame)
        label = "Lowpass"
        
    elif p_idx == 2:
        # Highpass Filter
        cutoff = rng.uniform(600, 5000)
        b, a = scipy.signal.butter(4, cutoff / (sr / 2.0), btype='high')
        perturbed = scipy.signal.filtfilt(b, a, frame)
        label = "Highpass"
        
    elif p_idx == 3:
        # Hard Clipping
        thresh = rng.uniform(0.02, 0.40)
        perturbed = np.clip(frame, -thresh, thresh)
        label = "Clipping"
        
    elif p_idx == 4:
        # Reverberation (Comb Filter feedback)
        delay = rng.integers(100, 1000)
        feedback = rng.uniform(0.30, 0.85)
        for i in range(delay, win):
            perturbed[i] += feedback * perturbed[i - delay]
        label = "Reverberation"
        
    elif p_idx == 5:
        # Harmonic Stripping
        subsets = [
            [1],          # Fundamental only
            [1, 3, 5],    # Odd harmonics only
            [2, 4],       # Even harmonics only
            [3, 4, 5],    # Upper harmonics only
        ]
        harmonics_subset = subsets[rng.choice(len(subsets))]
        t = np.arange(win) / sr
        perturbed = np.zeros(win)
        for h in harmonics_subset:
            perturbed += (1.0 / h) * np.sin(2 * np.pi * h * 220.0 * t)
        env = np.ones(win)
        fade = int(0.02 * sr)
        env[:fade] = np.linspace(0.0, 1.0, fade)
        env[-fade:] = np.linspace(1.0, 0.0, fade)
        perturbed *= env
        label = "Harmonic Stripping"
        
    elif p_idx == 6:
        # Transient Smearing (Moving Average)
        length = rng.integers(5, 60)
        perturbed = np.convolve(frame, np.ones(length) / length, mode='same')
        label = "Transient Smearing"
        
    elif p_idx == 7:
        # Frequency Modulation (FM Jitter)
        fm_rate = rng.uniform(20.0, 60.0)
        fm_depth = rng.uniform(10.0, 60.0)
        t = np.arange(win) / sr
        phase = 2 * np.pi * (220.0 * t + (fm_depth / fm_rate) * np.sin(2 * np.pi * fm_rate * t))
        perturbed = np.zeros(win)
        for k in range(1, 6):
            perturbed += (1.0 / k) * np.sin(k * phase)
        env = np.ones(win)
        fade = int(0.02 * sr)
        env[:fade] = np.linspace(0.0, 1.0, fade)
        env[-fade:] = np.linspace(1.0, 0.0, fade)
        perturbed *= env
        label = "FM Jitter"

    # Normalise RMS energy to clean frame RMS
    clean_rms = np.sqrt(np.mean(frame**2))
    perturbed_rms = np.sqrt(np.mean(perturbed**2))
    if perturbed_rms > 1e-9:
        perturbed *= (clean_rms / perturbed_rms)
        
    return perturbed, label


# ---------------------------------------------------------------------------
# Descriptor Extraction (11 Features)
# ---------------------------------------------------------------------------

def estimate_pitch_acf(acf: np.ndarray, sr: int) -> tuple[float, int]:
    min_lag = int(sr / 1000); max_lag = int(sr / 80)
    if max_lag > len(acf): max_lag = len(acf)
    acf_range = acf[min_lag:max_lag]
    lag_idx = np.argmax(acf_range) + min_lag
    return float(sr / lag_idx), int(lag_idx)


def estimate_pitch_cepstrum(cep: np.ndarray, sr: int) -> tuple[float, int]:
    min_lag = int(sr / 1000); max_lag = int(sr / 80)
    if max_lag > len(cep): max_lag = len(cep)
    cep_range = np.abs(cep[min_lag:max_lag])
    q_idx = np.argmax(cep_range) + min_lag
    return float(sr / q_idx), int(q_idx)


def estimate_pitch_stft(mag: np.ndarray, sr: int, n_fft: int = 2048) -> tuple[float, int]:
    freqs = np.fft.rfftfreq(n_fft, d=1/sr)
    min_bin = np.argmin(np.abs(freqs - 80)); max_bin = np.argmin(np.abs(freqs - 1000))
    bin_idx = np.argmax(mag[min_bin:max_bin]) + min_bin
    return float(freqs[bin_idx]), int(bin_idx)


def extract_descriptors(frame: np.ndarray, acf: np.ndarray, cep: np.ndarray,
                        mag: np.ndarray, sr: int, n_fft: int = 2048) -> list[float]:
    min_lag = int(sr / 1000); max_lag = int(sr / 80)
    freqs = np.fft.rfftfreq(n_fft, d=1/sr)
    min_bin = np.argmin(np.abs(freqs - 80)); max_bin = np.argmin(np.abs(freqs - 1000))
    min_q = min_lag; max_q = max_lag

    # 1. Spectral Entropy
    spec_range = mag[min_bin:max_bin]
    spec_sum = np.sum(spec_range)
    if spec_sum > 1e-9:
        p = spec_range / spec_sum
        p = np.clip(p, 1e-12, 1.0)
        spec_entropy = -np.sum(p * np.log2(p)) / np.log2(len(p))
    else:
        spec_entropy = 1.0

    # 2. Spectral Flatness
    if spec_sum > 1e-9:
        log_mean = np.mean(np.log(spec_range + 1e-12))
        spec_flatness = np.exp(log_mean) / (np.mean(spec_range) + 1e-12)
    else:
        spec_flatness = 1.0

    stft_p, stft_peak_idx = estimate_pitch_stft(mag, sr, n_fft)

    # 3. STFT Peak Strength
    stft_peak_strength = mag[stft_peak_idx] / (spec_sum + 1e-12)

    # 4. STFT Peak Prominence
    stft_peak_prominence = max(0.0, float((mag[stft_peak_idx] - np.mean(spec_range)) / (mag[stft_peak_idx] + 1e-12)))

    acf_p, acf_peak_idx = estimate_pitch_acf(acf, sr)

    # 5. ACF Peak Strength
    acf_peak_strength = acf[acf_peak_idx] / (acf[0] + 1e-12)

    # 6. ACF Peak Prominence
    acf_range = acf[min_lag:max_lag]
    acf_peak_prominence = float(np.clip((acf[acf_peak_idx] - np.mean(acf_range)) / (np.max(acf_range) - np.min(acf_range) + 1e-10), 0.0, 1.0))

    # 7. Cepstrum DC Coefficient c0
    cep_c0 = float(cep[0])

    cep_p, cep_peak_idx = estimate_pitch_cepstrum(cep, sr)

    # 8. Cepstral Peak Strength
    cep_peak_strength = float(np.abs(cep[cep_peak_idx]) / (np.abs(cep_c0) + 1e-10))

    # 9. Cepstral Peak Prominence
    cep_range = np.abs(cep[min_q:max_q])
    cep_peak_prominence = float(np.clip((np.abs(cep[cep_peak_idx]) - np.mean(cep_range)) / (np.max(cep_range) - np.min(cep_range) + 1e-10), 0.0, 1.0))

    # 10. Zero Crossing Rate (ZCR)
    zcr = float(np.mean(np.abs(np.diff(np.sign(frame)))) / 2.0)

    # 11. Frame Log Energy
    frame_log_energy = float(np.log(np.mean(frame**2) + 1e-10))

    return [
        spec_entropy, spec_flatness, stft_peak_strength, stft_peak_prominence,
        acf_peak_strength, acf_peak_prominence, cep_c0, cep_peak_strength,
        cep_peak_prominence, zcr, frame_log_energy
    ]


# ---------------------------------------------------------------------------
# Cosine Similarity Helper
# ---------------------------------------------------------------------------

def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    denom = (np.linalg.norm(v1) * np.linalg.norm(v2)) + 1e-10
    return float(np.dot(v1, v2) / denom)


# ---------------------------------------------------------------------------
# Principal Component Analysis (Pure NumPy)
# ---------------------------------------------------------------------------

def run_pca(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Standardises X and projects to 2D using SVD.
    Returns (X_pca, explained_variance_ratios, PC_components).
    """
    # Standardise
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0) + 1e-8
    X_std = (X - mu) / sigma
    
    # SVD
    U, S, Vt = np.linalg.svd(X_std, full_matrices=False)
    
    # Project to top 2 components
    X_pca = X_std @ Vt[:2].T
    
    # Explained variance ratios
    var_exp = (S ** 2) / np.sum(S ** 2)
    
    return X_pca, var_exp[:2], Vt[:2]


# ---------------------------------------------------------------------------
# K-Means Clustering (Pure NumPy)
# ---------------------------------------------------------------------------

def run_kmeans(X: np.ndarray, k: int, max_iters: int = 200, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """
    K-Means clustering in 2D.
    Returns (labels, centroids).
    """
    rng = np.random.default_rng(seed)
    centroids = X[rng.choice(X.shape[0], k, replace=False)]
    
    for _ in range(max_iters):
        # Distances between all points and centroids
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        
        # Recalculate centroids
        new_centroids = np.array([
            X[labels == j].mean(axis=0) if np.sum(labels == j) > 0 else centroids[j]
            for j in range(k)
        ])
        
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
        
    return labels, centroids


# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------

def run():
    print("=" * 60)
    print("EXPERIMENT 027 — FAILURE MANIFOLD MAPPING")
    print("=" * 60)

    sr = 22050
    WIN = 1024
    N_samples = 2000

    print("Generating reference clean harmonic stack...")
    clean_frame = synthesise_reference_frame(sr, WIN, f0=220.0)
    
    # Clean representations
    acf_clean = compute_acf(clean_frame)
    cep_clean = compute_cepstrum(clean_frame)
    mag_clean = compute_stft(clean_frame, sr)
    if mag_clean.ndim > 1: mag_clean = np.mean(mag_clean, axis=1)

    print(f"Generating {N_samples} random perturbed frames...")
    
    X_features = []
    similarities = []
    labels_type = []
    
    rng = np.random.default_rng(42)
    
    for idx in range(N_samples):
        p_idx = idx % 8
        perturbed, p_label = apply_random_perturbation(clean_frame, sr, p_idx, rng)
        
        # Compute representations
        acf_p = compute_acf(perturbed)
        cep_p = compute_cepstrum(perturbed)
        mag_p = compute_stft(perturbed, sr)
        if mag_p.ndim > 1: mag_p = np.mean(mag_p, axis=1)
        
        # Cosine similarity to clean
        sim_stft = cosine_similarity(mag_clean, mag_p)
        sim_acf  = cosine_similarity(acf_clean, acf_p)
        sim_cep  = cosine_similarity(cep_clean, cep_p)
        
        # Descriptors
        feats = extract_descriptors(perturbed, acf_p, cep_p, mag_p, sr, 2 * (len(mag_p)-1))
        
        X_features.append(feats)
        similarities.append([sim_stft, sim_acf, sim_cep])
        labels_type.append(p_label)

    X = np.array(X_features)
    similarities = np.array(similarities)
    labels_type = np.array(labels_type)
    
    print("Extracting PCA components on the 11-dimensional failure space...")
    X_pca, var_exp, Vt = run_pca(X)
    print(f"Explained Variance Ratio -> PC1: {var_exp[0]:.2%}, PC2: {var_exp[1]:.2%}")
    print(f"Total variance explained by top 2 PCs: {np.sum(var_exp):.2%}\n")
    
    print("Running K-Means clustering (k=5) in PCA projection space...")
    cluster_labels, centroids = run_kmeans(X_pca, k=5, seed=42)
    print("Clustering complete.\n")

    # Dynamic Cluster Auto-Labeling and Profiling
    cluster_profiles = {}
    print(f"{'Cluster':<10}  {'Size':<5}  {'STFT Sim':<10}  {'ACF Sim':<10}  {'Cep Sim':<10}  {'ZCR':<7}  {'Entropy':<8}  {'Discovered Profile Label'}")
    print("-" * 102)
    
    for j in range(5):
        mask = (cluster_labels == j)
        size = np.sum(mask)
        if size == 0: continue
        
        # Average similarities
        m_sim = similarities[mask].mean(axis=0)
        # Average key features (SF, SE, ZCR)
        m_feats = X[mask].mean(axis=0)
        # descriptors: 0=entropy, 1=flatness, 9=zcr
        m_ent = m_feats[0]
        m_zcr = m_feats[9]

        # Auto-labeling logic
        if m_sim[0] > 0.85 and m_sim[1] > 0.85 and m_sim[2] > 0.85:
            profile_label = "Healthy / Low-degradation Region"
        elif m_sim[0] < 0.40 and m_sim[1] < 0.40 and m_sim[2] < 0.40:
            profile_label = "Noise / Stochastic Collapse"
        elif m_sim[2] < 0.35 and m_sim[1] > 0.65:
            profile_label = "Harmonic / Periodicity Collapse"
        elif m_sim[0] < 0.40 and m_sim[2] < 0.40 and m_sim[1] > 0.50:
            profile_label = "Periodicity-Only (High LP/Clipping) Collapse"
        elif m_zcr > 0.15 and m_sim[1] < 0.50:
            profile_label = "Transient / High-frequency Smearing Collapse"
        elif m_sim[1] < 0.35 and m_sim[0] > 0.65:
            profile_label = "Spectral-Only (High Jitter/FM) Collapse"
        else:
            profile_label = "Mixed Degradation boundary"
            
        cluster_profiles[j] = {
            "size": size,
            "sims": m_sim,
            "label": profile_label,
            "zcr": m_zcr,
            "entropy": m_ent
        }
        
        print(f"{j:<10}  {size:<5}  {m_sim[0]:<10.3f}  {m_sim[1]:<10.3f}  {m_sim[2]:<10.3f}  {m_zcr:<7.3f}  {m_ent:<8.3f}  {profile_label}")
    print()

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    print("Generating failure manifold maps visualization...")
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.25)
    
    # 1. Cluster Scatter Plot
    ax_scatter = fig.add_subplot(gs[0, 0])
    scatter_colors = ["#377eb8", "#e7298a", "#a6d854", "#ff7f00", "#984ea3"]
    for j in range(5):
        mask = (cluster_labels == j)
        ax_scatter.scatter(X_pca[mask, 0], X_pca[mask, 1], s=12,
                           color=scatter_colors[j], alpha=0.7,
                           label=f"C{j}: {cluster_profiles[j]['label']}")
    ax_scatter.scatter(centroids[:, 0], centroids[:, 1], s=120, color="white", marker="X", edgecolors="black", label="Centroids")
    ax_scatter.set_title(f"Discovered Clusters in PCA Projection space (Exp variance: {np.sum(var_exp):.1%})", fontsize=11, fontweight="bold")
    ax_scatter.set_xlabel("PC1")
    ax_scatter.set_ylabel("PC2")
    ax_scatter.legend(fontsize=7, loc="lower left")
    ax_scatter.grid(True, alpha=0.08)

    # 2. STFT Similarity Heatmap
    ax_stft = fig.add_subplot(gs[0, 1])
    sc_stft = ax_stft.scatter(X_pca[:, 0], X_pca[:, 1], s=8, c=similarities[:, 0], cmap="viridis", alpha=0.7)
    fig.colorbar(sc_stft, ax=ax_stft, label="STFT Cosine Similarity")
    ax_stft.set_title("STFT Failure Map (Low similarity = Spectral Collapse)", fontsize=11, fontweight="bold")
    ax_stft.grid(True, alpha=0.08)

    # 3. ACF Similarity Heatmap
    ax_acf = fig.add_subplot(gs[1, 0])
    sc_acf = ax_acf.scatter(X_pca[:, 0], X_pca[:, 1], s=8, c=similarities[:, 1], cmap="magma", alpha=0.7)
    fig.colorbar(sc_acf, ax=ax_acf, label="ACF Cosine Similarity")
    ax_acf.set_title("ACF Failure Map (Low similarity = Periodicity Collapse)", fontsize=11, fontweight="bold")
    ax_acf.grid(True, alpha=0.08)

    # 4. Cepstrum Similarity Heatmap
    ax_cep = fig.add_subplot(gs[1, 1])
    sc_cep = ax_cep.scatter(X_pca[:, 0], X_pca[:, 1], s=8, c=similarities[:, 2], cmap="plasma", alpha=0.7)
    fig.colorbar(sc_cep, ax=ax_cep, label="Cepstrum Cosine Similarity")
    ax_cep.set_title("Cepstrum Failure Map (Low similarity = Harmonic Collapse)", fontsize=11, fontweight="bold")
    ax_cep.grid(True, alpha=0.08)

    # 5. Profile Bar Chart
    ax_bar = fig.add_subplot(gs[2, :])
    x = np.arange(5)
    width = 0.22
    
    stft_means = [cluster_profiles[j]["sims"][0] for j in range(5)]
    acf_means  = [cluster_profiles[j]["sims"][1] for j in range(5)]
    cep_means  = [cluster_profiles[j]["sims"][2] for j in range(5)]
    
    b1 = ax_bar.bar(x - width, stft_means, width, label="STFT Sim", color="#a6d854", alpha=0.85)
    b2 = ax_bar.bar(x, acf_means, width, label="ACF Sim", color="#377eb8", alpha=0.85)
    b3 = ax_bar.bar(x + width, cep_means, width, label="Cepstrum Sim", color="#e7298a", alpha=0.85)
    
    for bars in [b1, b2, b3]:
        for bar in bars:
            height = bar.get_height()
            ax_bar.text(bar.get_x() + bar.get_width()/2, height + 0.02,
                        f"{height:.2f}", ha="center", va="bottom", fontsize=7, color="white")
            
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([f"C{j}: {cluster_profiles[j]['label'][:32]}..." for j in range(5)], fontsize=8)
    ax_bar.set_ylabel("Mean Cosine Similarity")
    ax_bar.set_ylim(0, 1.2)
    ax_bar.set_title("Mean Representation Similarity across Discovered Collapse Clusters", fontsize=11, fontweight="bold")
    ax_bar.legend(fontsize=9, loc="upper right")
    ax_bar.grid(True, alpha=0.08, axis="y")

    fig.suptitle("Experiment 027 — Failure Manifold Mapping: Discovering Task-Independent Collapse Clusters", fontsize=14, fontweight="bold", y=0.98)
    
    out_path = os.path.join(project_root, "results", "exp027_failure_manifold.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    run()
