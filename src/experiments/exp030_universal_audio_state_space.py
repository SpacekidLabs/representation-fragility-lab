"""
Experiment 030 — Universal Audio State Space
============================================
Forget representation similarities and failure metrics.
Builds the manifold using ONLY 10 pure physical descriptors from a diverse corpus of 10 signal classes:
Speech, Vocals, Piano, Guitar, Drums, Environmental (Robin), Synths, FM, Granular, Noise.
Projects degradation trajectories to determine if the manifold is a projection of audio itself or representation-specific.
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
    extract_high_energy_frames,
    run_pca
)
from src.experiments.exp029_failure_trajectories import run_dbscan

# ---------------------------------------------------------------------------
# Synth Generators
# ---------------------------------------------------------------------------

def synthesize_synth_waves(f0: float, sr: int, win: int, count: int) -> list[np.ndarray]:
    frames = []
    rng = np.random.default_rng(42)
    t = np.arange(win) / sr
    while len(frames) < count:
        wave_type = rng.choice(["saw", "square"])
        if wave_type == "saw":
            frame = scipy.signal.sawtooth(2 * np.pi * f0 * t)
        else:
            frame = scipy.signal.square(2 * np.pi * f0 * t)
        frame += rng.normal(0, 0.01, win)
        frame /= np.max(np.abs(frame)) + 1e-9
        frames.append(frame)
    return frames

def synthesize_fm_waves(f_carrier: float, sr: int, win: int, count: int) -> list[np.ndarray]:
    frames = []
    rng = np.random.default_rng(42)
    t = np.arange(win) / sr
    while len(frames) < count:
        f_mod = rng.uniform(20.0, 100.0)
        index = rng.uniform(1.0, 10.0)
        modulator = index * np.sin(2 * np.pi * f_mod * t)
        frame = np.sin(2 * np.pi * f_carrier * t + modulator)
        frame /= np.max(np.abs(frame)) + 1e-9
        frames.append(frame)
    return frames

def synthesize_granular_textures(sr: int, win: int, count: int) -> list[np.ndarray]:
    frames = []
    rng = np.random.default_rng(42)
    t = np.arange(win) / sr
    while len(frames) < count:
        frame = np.zeros(win)
        num_grains = rng.integers(5, 15)
        for _ in range(num_grains):
            f_grain = rng.uniform(150.0, 800.0)
            t_offset = rng.uniform(0, 0.02)
            grain_win = rng.uniform(0.01, 0.05)
            env = np.exp(-((t - t_offset) / grain_win)**2)
            frame += env * np.sin(2 * np.pi * f_grain * t)
        frame /= np.max(np.abs(frame)) + 1e-9
        frames.append(frame)
    return frames

def generate_noise_frames(win: int, count: int) -> list[np.ndarray]:
    frames = []
    rng = np.random.default_rng(42)
    while len(frames) < count:
        noise_type = rng.choice(["white", "pink"])
        if noise_type == "white":
            frame = rng.normal(0, 0.3, win)
        else:
            white = rng.normal(0, 0.3, win)
            b = [1.0]; a = [1.0, -0.99]
            frame = scipy.signal.lfilter(b, a, white)
        frame /= np.max(np.abs(frame)) + 1e-9
        frames.append(frame)
    return frames

# ---------------------------------------------------------------------------
# 10 Physical Descriptors Extraction
# ---------------------------------------------------------------------------

def extract_10_physical_descriptors(frame: np.ndarray, sr: int, n_fft: int = 2048) -> list[float]:
    # Compute representations
    acf = compute_acf(frame)
    mag = compute_stft(frame, sr)
    if mag.ndim > 1: mag = np.mean(mag, axis=1)

    # 1. Spectral Entropy
    mag_sum = np.sum(mag)
    if mag_sum > 1e-12:
        p = mag / mag_sum
        p = np.clip(p, 1e-12, 1.0)
        entropy = float(-np.sum(p * np.log2(p)) / np.log2(len(p)))
    else:
        entropy = 1.0

    # 2. Spectral Flatness
    if mag_sum > 1e-12:
        log_mean = np.mean(np.log(mag + 1e-12))
        flatness = float(np.exp(log_mean) / (np.mean(mag) + 1e-12))
    else:
        flatness = 1.0

    # 3. ZCR
    zcr = float(np.mean(np.abs(np.diff(np.sign(frame)))) / 2.0)

    # 4. Harmonic Ratio
    min_lag = int(sr / 1000)
    max_lag = int(sr / 80)
    if max_lag > len(acf): max_lag = len(acf)
    lag_idx = np.argmax(acf[min_lag:max_lag]) + min_lag
    harmonic_ratio = float(acf[lag_idx] / (acf[0] + 1e-12))

    # 5. Crest Factor
    rms = np.sqrt(np.mean(frame**2))
    crest_factor = float(np.max(np.abs(frame)) / (rms + 1e-12))

    # 6. Periodicity (ACF Prominence)
    acf_range = acf[min_lag:max_lag]
    periodicity = float(np.clip((acf[lag_idx] - np.mean(acf_range)) / (np.max(acf_range) - np.min(acf_range) + 1e-10), 0.0, 1.0))

    # 7. Spectral Rolloff
    freqs = np.fft.rfftfreq(n_fft, d=1/sr)
    cum_sum = np.cumsum(mag)
    total = cum_sum[-1]
    if total > 1e-12:
        idx = np.where(cum_sum >= 0.85 * total)[0][0]
        rolloff = float(freqs[idx])
    else:
        rolloff = 0.0

    # 8. Sparsity (Hoyer)
    n_bins = len(mag)
    l1 = np.sum(np.abs(mag))
    l2 = np.sqrt(np.sum(mag**2))
    if l2 > 1e-12:
        ratio = l1 / l2
        sparsity = float((np.sqrt(n_bins) - ratio) / (np.sqrt(n_bins) - 1.0 + 1e-12))
    else:
        sparsity = 0.0

    # 9. Transientness (Kurtosis)
    mu = np.mean(frame)
    std = np.std(frame) + 1e-8
    transientness = float(np.mean((frame - mu)**4) / (std**4))

    # 10. Modulation Descriptor
    subdivisions = np.split(frame, 4)
    rms_vals = [np.sqrt(np.mean(sub**2)) for sub in subdivisions]
    modulation = float(np.std(rms_vals))

    return [
        entropy, flatness, zcr, harmonic_ratio, crest_factor,
        periodicity, rolloff, sparsity, transientness, modulation
    ]

# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------

def run():
    print("=" * 75)
    print("EXPERIMENT 030 — UNIVERSAL AUDIO STATE SPACE")
    print("=" * 75)
    sr = 22050
    win = 1024
    N = 200  # 200 frames per category
    N_total = N * 10  # 2000 total frames

    print("\nLoading and synthesizing diverse signal categories (10 classes)...")
    
    # 1. Speech
    print("Loading Speech ('libri1')...")
    y_sp, _ = librosa.load(librosa.example('libri1'), sr=sr)
    speech_frames = extract_high_energy_frames(y_sp, win, N)

    # 2. Vocals
    print("Loading Vocals ('Clean_vocal.wav')...")
    y_voc, _ = librosa.load(os.path.join(project_root, "Clean_vocal.wav"), sr=sr)
    vocals_frames = extract_high_energy_frames(y_voc, win, N)

    # 3. Piano
    print("Loading Piano ('pistachio')...")
    y_pn, _ = librosa.load(librosa.example('pistachio'), sr=sr)
    piano_frames = extract_high_energy_frames(y_pn, win, N)

    # 4. Drums
    print("Loading Drums ('choice')...")
    y_dr, _ = librosa.load(librosa.example('choice'), sr=sr)
    drums_frames = extract_high_energy_frames(y_dr, win, N)

    # 5. Guitar
    print("Synthesizing plucked guitar notes (Karplus-Strong)...")
    guitar_frames = []
    rng = np.random.default_rng(42)
    f0s = [82.4, 110.0, 146.8, 196.0, 220.0, 329.6]
    while len(guitar_frames) < N:
        f0 = rng.choice(f0s)
        pluck = synthesize_karplus_strong(f0, sr, int(1.2 * sr))
        guitar_frames.extend(extract_high_energy_frames(pluck, win, 20))
    guitar_frames = guitar_frames[:N]

    # 6. Environmental (Robin Bird Whistles)
    print("Loading Environmental example ('robin')...")
    y_rb, _ = librosa.load(librosa.example('robin'), sr=sr)
    robin_frames = extract_high_energy_frames(y_rb, win, N)

    # 7. Synths
    print("Synthesizing Synth waves (Saw/Square)...")
    synth_frames = []
    while len(synth_frames) < N:
        synth_frames.extend(synthesize_synth_waves(rng.choice(f0s), sr, win, 50))
    synth_frames = synth_frames[:N]

    # 8. FM Tone
    print("Synthesizing FM waves...")
    fm_frames = []
    while len(fm_frames) < N:
        fm_frames.extend(synthesize_fm_waves(rng.choice(f0s), sr, win, 50))
    fm_frames = fm_frames[:N]

    # 9. Granular
    print("Synthesizing Granular textures...")
    granular_frames = []
    while len(granular_textures := synthesize_granular_textures(sr, win, N)) < N:
        pass
    granular_frames = granular_textures[:N]

    # 10. Noise
    print("Generating Noise waves (White/Pink)...")
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

    # Verify sizes
    print("\nSignal Classes Loaded:")
    for k, v in all_corpora.items():
        print(f" - {k:<15}: {len(v)} frames")

    # Extract physical descriptors
    print("\nExtracting 10 physical descriptors across all 2,000 frames...")
    X_features = []
    corpus_labels = []
    
    for c_name, frames in all_corpora.items():
        for frame in frames:
            feats = extract_10_physical_descriptors(frame, sr, 2048)
            X_features.append(feats)
            corpus_labels.append(c_name)

    X = np.array(X_features)
    corpus_labels = np.array(corpus_labels)

    # Fit PCA Projection
    print("\nConstructing Universal Audio State Space (PCA)...")
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0) + 1e-8
    X_std = (X - mu) / sigma
    
    X_pca, var_exp, Vt = run_pca(X)
    print(f"PCA explained variance ratios -> PC1: {var_exp[0]:.2%}, PC2: {var_exp[1]:.2%}, Total: {np.sum(var_exp):.2%}")

    # Run DBSCAN Clustering
    print("\nRunning DBSCAN density verification on Audio State Space...")
    X_pca_std = (X_pca - np.mean(X_pca, axis=0)) / np.std(X_pca, axis=0)
    db_labels = run_dbscan(X_pca_std, eps=0.35, min_samples=15)
    
    unique_clusters = np.unique(db_labels)
    print(f"DBSCAN found {len(unique_clusters) - (1 if -1 in db_labels else 0)} clusters + noise.")

    # ---------------------------------------------------------------------------
    # Axis Correlations
    # ---------------------------------------------------------------------------
    print("\nPC1 & PC2 Axis Correlation to Physical Properties:")
    x_lbls = ["Spectral Entropy", "Spectral Flatness", "ZCR", "Harmonic Ratio", 
              "Crest Factor", "Periodicity", "Spectral Rolloff", "Sparsity", "Kurtosis", "Modulation"]
    
    from src.experiments.exp029_failure_trajectories import pearson_correlation
    print(f"{'Physical Descriptor':<20} | {'PC1 (r)':<8} | {'PC2 (r)':<8}")
    print("-" * 44)
    for idx_feat, name in enumerate(x_lbls):
        r1 = pearson_correlation(X_pca[:, 0], X[:, idx_feat])
        r2 = pearson_correlation(X_pca[:, 1], X[:, idx_feat])
        print(f"{name:<20} | {r1:<8.3f} | {r2:<8.3f}")

    # ---------------------------------------------------------------------------
    # Project Degradation Trajectories (Deciding Test)
    # ---------------------------------------------------------------------------
    print("\nProjecting degradation trajectories from Exp 029 onto the Audio State Space...")
    steps = 50
    
    # Path 1: Vocal Noise Path
    vocal_clean = vocals_frames[0]
    path_1_coords = []
    for k in range(steps):
        sig = 1.0 * (k / (steps - 1))
        y_pert = vocal_clean + rng.normal(0, sig, win)
        y_pert *= (np.sqrt(np.mean(vocal_clean**2)) / np.sqrt(np.mean(y_pert**2) + 1e-12))
        feats = extract_10_physical_descriptors(y_pert, sr)
        feats_std = (np.array(feats) - mu) / sigma
        path_1_coords.append(feats_std @ Vt.T)
    path_1_coords = np.array(path_1_coords)

    # Path 2: Guitar Filtering Path
    guitar_clean = guitar_frames[0]
    path_2_coords = []
    for k in range(steps):
        fc = 2000.0 - 1850.0 * (k / (steps - 1))
        b, a = scipy.signal.butter(4, fc / (sr / 2.0), btype='low')
        y_pert = scipy.signal.filtfilt(b, a, guitar_clean)
        y_pert *= (np.sqrt(np.mean(guitar_clean**2)) / np.sqrt(np.mean(y_pert**2) + 1e-12))
        feats = extract_10_physical_descriptors(y_pert, sr)
        feats_std = (np.array(feats) - mu) / sigma
        path_2_coords.append(feats_std @ Vt.T)
    path_2_coords = np.array(path_2_coords)

    # Path 3: Piano Saturation Path
    piano_clean = piano_frames[0]
    path_3_coords = []
    for k in range(steps):
        drive = 1.0 + 19.0 * (k / (steps - 1))
        y_pert = np.tanh(drive * piano_clean) / np.tanh(drive)
        y_pert *= (np.sqrt(np.mean(piano_clean**2)) / np.sqrt(np.mean(y_pert**2) + 1e-12))
        feats = extract_10_physical_descriptors(y_pert, sr)
        feats_std = (np.array(feats) - mu) / sigma
        path_3_coords.append(feats_std @ Vt.T)
    path_3_coords = np.array(path_3_coords)

    # ---------------------------------------------------------------------------
    # Plotting
    # ---------------------------------------------------------------------------
    print("\nGenerating Universal Audio State Space plots...")
    fig = plt.figure(figsize=(22, 15))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25)
    
    # 1. State Space colored by Signal Classes
    ax_class = fig.add_subplot(gs[0, 0])
    cmap = plt.get_cmap("tab10")
    for idx_c, (c_name, _) in enumerate(all_corpora.items()):
        mask = (corpus_labels == c_name)
        ax_class.scatter(X_pca[mask, 0], X_pca[mask, 1], s=12, color=cmap(idx_c), alpha=0.7, label=c_name)
    ax_class.set_title("Universal Audio State Space\n(Colored by Signal Category)", fontsize=11, fontweight="bold")
    ax_class.set_xlabel("PC1: Order ↔ Disorder")
    ax_class.set_ylabel("PC2: Harmonic ↔ Transient")
    ax_class.legend(fontsize=8, loc="upper right")
    ax_class.grid(True, alpha=0.08)

    # 2. State Space colored by DBSCAN Clusters
    ax_db = fig.add_subplot(gs[0, 1])
    scatter_colors = ["#377eb8", "#e7298a", "#a6d854", "#ff7f00", "#984ea3"]
    for cid in unique_clusters:
        mask = (db_labels == cid)
        col = "gray" if cid == -1 else scatter_colors[cid % len(scatter_colors)]
        lbl = "Noise / Outliers" if cid == -1 else f"Cluster {cid}"
        ax_db.scatter(X_pca[mask, 0], X_pca[mask, 1], s=12, color=col, alpha=0.6, label=lbl)
    ax_db.set_title("DBSCAN Density Clusters on Audio State Space", fontsize=11, fontweight="bold")
    ax_db.set_xlabel("PC1")
    ax_db.set_ylabel("PC2")
    ax_db.legend(fontsize=8, loc="lower left")
    ax_db.grid(True, alpha=0.08)

    # 3. Continuous Trajectories Flow Overlay
    ax_traj = fig.add_subplot(gs[1, :])
    # Scatter background database points lightly
    ax_traj.scatter(X_pca[:, 0], X_pca[:, 1], s=3, color="white", alpha=0.15, label="Universal Audio Frames")
    
    # Vocals Noise Path
    ax_traj.plot(path_1_coords[:, 0], path_1_coords[:, 1], color="#e7298a", linewidth=3.0, label="Vocals + Noise Path")
    ax_traj.scatter(path_1_coords[0, 0], path_1_coords[0, 1], marker="o", s=100, color="#e7298a", edgecolors="white", zorder=5)
    ax_traj.scatter(path_1_coords[-1, 0], path_1_coords[-1, 1], marker="X", s=120, color="#e7298a", edgecolors="white", zorder=5)
    for step_i in [10, 25, 40]:
        ax_traj.annotate("", xy=(path_1_coords[step_i+1, 0], path_1_coords[step_i+1, 1]),
                         xytext=(path_1_coords[step_i, 0], path_1_coords[step_i, 1]),
                         arrowprops=dict(arrowstyle="->", color="#e7298a", lw=2))

    # Guitar Lowpass Path
    ax_traj.plot(path_2_coords[:, 0], path_2_coords[:, 1], color="#ff7f00", linewidth=3.0, label="Guitar + Lowpass Path")
    ax_traj.scatter(path_2_coords[0, 0], path_2_coords[0, 1], marker="o", s=100, color="#ff7f00", edgecolors="white", zorder=5)
    ax_traj.scatter(path_2_coords[-1, 0], path_2_coords[-1, 1], marker="X", s=120, color="#ff7f00", edgecolors="white", zorder=5)
    for step_i in [10, 25, 40]:
        ax_traj.annotate("", xy=(path_2_coords[step_i+1, 0], path_2_coords[step_i+1, 1]),
                         xytext=(path_2_coords[step_i, 0], path_2_coords[step_i, 1]),
                         arrowprops=dict(arrowstyle="->", color="#ff7f00", lw=2))

    # Piano Saturation Path
    ax_traj.plot(path_3_coords[:, 0], path_3_coords[:, 1], color="#984ea3", linewidth=3.0, label="Piano + Saturation Path")
    ax_traj.scatter(path_3_coords[0, 0], path_3_coords[0, 1], marker="o", s=100, color="#984ea3", edgecolors="white", zorder=5)
    ax_traj.scatter(path_3_coords[-1, 0], path_3_coords[-1, 1], marker="X", s=120, color="#984ea3", edgecolors="white", zorder=5)
    for step_i in [10, 25, 40]:
        ax_traj.annotate("", xy=(path_3_coords[step_i+1, 0], path_3_coords[step_i+1, 1]),
                         xytext=(path_3_coords[step_i, 0], path_3_coords[step_i, 1]),
                         arrowprops=dict(arrowstyle="->", color="#984ea3", lw=2))

    ax_traj.set_title("Dynamical Degradation Sweeps Overlaid on the Universal Audio State Space", fontsize=11, fontweight="bold")
    ax_traj.set_xlabel("PC1: Order ↔ Disorder")
    ax_traj.set_ylabel("PC2: Harmonic ↔ Transient")
    ax_traj.legend(fontsize=9, loc="upper right")
    ax_traj.grid(True, alpha=0.08)

    # Annotated zones on the trajectory plot
    ax_traj.text(-2.0, -1.0, "Periodic Harmonic Region\n(Clean vocals, piano, synth)", color="green", weight="bold", fontsize=9, bbox=dict(facecolor='black', alpha=0.6, edgecolor='none'))
    ax_traj.text(2.0, 1.5, "Noise Collapse Region\n(Uncorrelated white/pink noise)", color="red", weight="bold", fontsize=9, bbox=dict(facecolor='black', alpha=0.6, edgecolor='none'))
    ax_traj.text(-1.5, 2.0, "Smooth Lowpass Region\n(Low cutoff filtering, fundamental sines)", color="orange", weight="bold", fontsize=9, bbox=dict(facecolor='black', alpha=0.6, edgecolor='none'))
    ax_traj.text(1.0, -1.8, "Transient / Overloaded Region\n(Saturated waves, drums, clicks)", color="purple", weight="bold", fontsize=9, bbox=dict(facecolor='black', alpha=0.6, edgecolor='none'))

    fig.suptitle("Experiment 030 — Universal Audio State Space:\nValidating the Geometric Reality of Representation Failure Trajectories on 10 Signal Classes", fontsize=14, fontweight="bold", y=0.98)
    
    out_path = os.path.join(project_root, "results", "exp030_universal_audio_state_space.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\nSaved universal state space plot: {out_path}")
    print("=" * 75)


if __name__ == "__main__":
    run()
