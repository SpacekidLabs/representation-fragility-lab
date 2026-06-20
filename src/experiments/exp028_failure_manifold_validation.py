"""
Experiment 028 — Failure Manifold Validation
=============================================
Stress-tests the failure manifold hypothesis to check if the 2D topology is a universal,
task-independent physical layout of representation collapse.

Tests Implemented:
1. New Representations: CQT, Wavelet CWT, Mel Spectrogram (17 descriptors total).
2. New Signals: Real vocals, speech (LibriSpeech), piano (ragtime), drums (drum+bass), guitar (Karplus-Strong pluck).
3. New Perturbations: Compression, Soft Saturation, Bitcrushing, MP3 Spectral Quantization (12 perturbations total).
4. Predictive Power: Trains polynomial Ridge regression (degree 2) on 2D coordinates to predict pitch/onset errors.
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

# Suppress librosa user warnings for short signals
warnings.filterwarnings('ignore', category=UserWarning)

from src.representations.acf import compute_acf
from src.representations.cepstrum import compute_cepstrum
from src.representations.stft import compute_stft

# ---------------------------------------------------------------------------
# Karplus-Strong Guitar Pluck Synthesizer
# ---------------------------------------------------------------------------

def synthesize_karplus_strong(f0: float, sr: int, duration_samples: int) -> np.ndarray:
    """
    Synthesize a plucked string (guitar-like) note using Karplus-Strong physical modeling.
    """
    N = int(sr / f0)
    # White noise excitation buffer
    ring_buffer = np.random.uniform(-1.0, 1.0, N)
    out = np.zeros(duration_samples)
    for i in range(duration_samples):
        val = ring_buffer[i % N]
        out[i] = val
        # First-order IIR feedback lowpass filter (averaging adjacent samples)
        next_val = 0.996 * 0.5 * (val + ring_buffer[(i + 1) % N])
        ring_buffer[i % N] = next_val
    return out

# ---------------------------------------------------------------------------
# Morlet Wavelet Continuous Wavelet Transform (Pure NumPy FFT-based)
# ---------------------------------------------------------------------------

def compute_morlet_cwt(frame: np.ndarray, n_scales: int = 64) -> np.ndarray:
    """
    Computes a Morlet continuous wavelet transform (CWT) magnitude vector.
    """
    n = len(frame)
    scales = np.linspace(2, 128, n_scales)
    w0 = 5.0
    x_fft = np.fft.fft(frame)
    freqs = np.fft.fftfreq(n) * 2 * np.pi
    cwt_matrix = np.zeros((len(scales), n))
    for i, s in enumerate(scales):
        # Morlet wavelet in frequency domain
        wavelet_fft = np.exp(-0.5 * (s * freqs - w0)**2)
        cwt_matrix[i, :] = np.abs(np.fft.ifft(x_fft * wavelet_fft))
    # Average across time to get a scale vector
    return np.mean(cwt_matrix, axis=1)

# ---------------------------------------------------------------------------
# Representation Wrappers
# ---------------------------------------------------------------------------

def compute_cqt_frame(frame: np.ndarray, sr: int) -> np.ndarray:
    # Pad to avoid edge effects and CQT warnings
    frame_pad = np.pad(frame, 1536, mode='reflect')
    c = librosa.cqt(frame_pad, sr=sr, n_bins=84, bins_per_octave=12)
    return np.mean(np.abs(c), axis=1)

def compute_mel_frame(frame: np.ndarray, sr: int) -> np.ndarray:
    m = librosa.feature.melspectrogram(y=frame, sr=sr, n_fft=1024, hop_length=256, n_mels=128)
    return np.mean(m, axis=1)

# ---------------------------------------------------------------------------
# Perturbations Engine (12 Classes)
# ---------------------------------------------------------------------------

def apply_random_perturbation(frame: np.ndarray, sr: int, p_idx: int, rng: np.random.Generator) -> tuple[np.ndarray, str]:
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
        # Reverberation (Comb filter)
        delay = rng.integers(100, 1000)
        feedback = rng.uniform(0.30, 0.85)
        for i in range(delay, win):
            perturbed[i] += feedback * perturbed[i - delay]
        label = "Reverberation"
        
    elif p_idx == 5:
        # Harmonic Stripping (Comb notch at 220 Hz)
        delay = int(sr / 220.0)
        for i in range(delay, win):
            perturbed[i] -= 0.95 * perturbed[i - delay]
        label = "Harmonic Stripping"
        
    elif p_idx == 6:
        # Transient Smearing (Moving Average)
        length = rng.integers(5, 60)
        perturbed = np.convolve(frame, np.ones(length) / length, mode='same')
        label = "Transient Smearing"
        
    elif p_idx == 7:
        # FM Jitter
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

    elif p_idx == 8:
        # Dynamic Range Compression
        thresh = rng.uniform(0.02, 0.15)
        ratio = rng.uniform(3.0, 10.0)
        env = np.abs(frame)
        gain = np.ones_like(frame)
        mask = env > thresh
        if np.any(mask):
            gain[mask] = thresh + (env[mask] - thresh) / ratio
            gain = np.where(env > 0, gain / env, 1.0)
            perturbed = frame * gain
        label = "Compression"

    elif p_idx == 9:
        # Soft Saturation (tanh)
        drive = rng.uniform(2.0, 8.0)
        perturbed = np.tanh(drive * frame) / np.tanh(drive)
        label = "Saturation"

    elif p_idx == 10:
        # Bitcrushing
        bits = rng.integers(2, 6)
        levels = 2**(bits - 1)
        perturbed = np.round(frame * levels) / levels
        label = "Bitcrushing"

    elif p_idx == 11:
        # MP3 Spectral Quantization
        keep_ratio = rng.uniform(0.05, 0.30)
        f_coef = np.fft.rfft(frame)
        mags = np.abs(f_coef)
        thresh = np.percentile(mags, (1.0 - keep_ratio) * 100)
        f_coef[mags < thresh] = 0.0
        # Coarse quantization
        f_coef = np.round(f_coef * 5.0) / 5.0
        perturbed = np.fft.irfft(f_coef, n=win)
        label = "MP3 Quantization"

    # Normalise RMS energy to clean frame RMS
    clean_rms = np.sqrt(np.mean(frame**2))
    perturbed_rms = np.sqrt(np.mean(perturbed**2))
    if perturbed_rms > 1e-9:
        perturbed *= (clean_rms / perturbed_rms)
        
    return perturbed, label

# ---------------------------------------------------------------------------
# Feature Extraction (17 descriptors)
# ---------------------------------------------------------------------------

def extract_17_descriptors(frame: np.ndarray, acf: np.ndarray, cep: np.ndarray,
                           mag: np.ndarray, cqt: np.ndarray, cwt: np.ndarray,
                           mel: np.ndarray, sr: int, n_fft: int = 2048) -> list[float]:
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

    # 3. STFT Peak Strength
    stft_peak_idx = np.argmax(mag[min_bin:max_bin]) + min_bin
    stft_peak_strength = mag[stft_peak_idx] / (spec_sum + 1e-12)

    # 4. STFT Peak Prominence
    stft_peak_prominence = max(0.0, float((mag[stft_peak_idx] - np.mean(spec_range)) / (mag[stft_peak_idx] + 1e-12)))

    # 5. ACF Peak Strength
    acf_range = acf[min_lag:max_lag]
    acf_peak_idx = np.argmax(acf_range) + min_lag
    acf_peak_strength = acf[acf_peak_idx] / (acf[0] + 1e-12)

    # 6. ACF Peak Prominence
    acf_peak_prominence = float(np.clip((acf[acf_peak_idx] - np.mean(acf_range)) / (np.max(acf_range) - np.min(acf_range) + 1e-10), 0.0, 1.0))

    # 7. Cepstrum DC Coefficient c0
    cep_c0 = float(cep[0])

    # 8. Cepstral Peak Strength
    cep_range = np.abs(cep[min_q:max_q])
    cep_peak_idx = np.argmax(cep_range) + min_q
    cep_peak_strength = float(np.abs(cep[cep_peak_idx]) / (np.abs(cep_c0) + 1e-10))

    # 9. Cepstral Peak Prominence
    cep_peak_prominence = float(np.clip((np.abs(cep[cep_peak_idx]) - np.mean(cep_range)) / (np.max(cep_range) - np.min(cep_range) + 1e-10), 0.0, 1.0))

    # 10. Zero Crossing Rate (ZCR)
    zcr = float(np.mean(np.abs(np.diff(np.sign(frame)))) / 2.0)

    # 11. Frame Log Energy
    frame_log_energy = float(np.log(np.mean(frame**2) + 1e-10))

    # 12. CQT Peak Strength
    cqt_sum = np.sum(cqt)
    cqt_peak_idx = np.argmax(cqt)
    cqt_peak_strength = cqt[cqt_peak_idx] / (cqt_sum + 1e-12)

    # 13. CQT Peak Prominence
    cqt_peak_prominence = float((cqt[cqt_peak_idx] - np.mean(cqt)) / (cqt[cqt_peak_idx] + 1e-12))

    # 15. Mel Spectrogram Entropy
    mel_sum = np.sum(mel)
    if mel_sum > 1e-9:
        p_mel = mel / mel_sum
        p_mel = np.clip(p_mel, 1e-12, 1.0)
        mel_entropy = -np.sum(p_mel * np.log2(p_mel)) / np.log2(len(p_mel))
    else:
        mel_entropy = 1.0

    # 16. Mel Spectrogram Flatness
    if mel_sum > 1e-9:
        log_mean_mel = np.mean(np.log(mel + 1e-12))
        mel_flatness = np.exp(log_mean_mel) / (np.mean(mel) + 1e-12)
    else:
        mel_flatness = 1.0

    # 17. Wavelet Peak Strength
    cwt_sum = np.sum(cwt)
    cwt_peak_idx = np.argmax(cwt)
    cwt_peak_strength = cwt[cwt_peak_idx] / (cwt_sum + 1e-12)

    # 18. Wavelet Peak Prominence
    cwt_peak_prominence = float((cwt[cwt_peak_idx] - np.mean(cwt)) / (cwt[cwt_peak_idx] + 1e-12))

    return [
        spec_entropy, spec_flatness, stft_peak_strength, stft_peak_prominence,
        acf_peak_strength, acf_peak_prominence, cep_c0, cep_peak_strength,
        cep_peak_prominence, zcr, frame_log_energy,
        cqt_peak_strength, cqt_peak_prominence,
        mel_entropy, mel_flatness,
        cwt_peak_strength, cwt_peak_prominence
    ]

# ---------------------------------------------------------------------------
# Cosine Similarity
# ---------------------------------------------------------------------------

def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    denom = (np.linalg.norm(v1) * np.linalg.norm(v2)) + 1e-10
    return float(np.dot(v1, v2) / denom)

# ---------------------------------------------------------------------------
# Pure NumPy PCA
# ---------------------------------------------------------------------------

def run_pca(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0) + 1e-8
    X_std = (X - mu) / sigma
    U, S, Vt = np.linalg.svd(X_std, full_matrices=False)
    X_pca = X_std @ Vt[:2].T
    var_exp = (S ** 2) / np.sum(S ** 2)
    return X_pca, var_exp[:2], Vt[:2]

# ---------------------------------------------------------------------------
# Pure NumPy K-Means
# ---------------------------------------------------------------------------

def run_kmeans(X: np.ndarray, k: int, max_iters: int = 200, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    centroids = X[rng.choice(X.shape[0], k, replace=False)]
    for _ in range(max_iters):
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        new_centroids = np.array([
            X[labels == j].mean(axis=0) if np.sum(labels == j) > 0 else centroids[j]
            for j in range(k)
        ])
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
    return labels, centroids

# ---------------------------------------------------------------------------
# Pitch Estimators (For Pitch-Tracker Failure Target)
# ---------------------------------------------------------------------------

def estimate_pitch_stft(mag: np.ndarray, sr: int, n_fft: int = 2048) -> float:
    freqs = np.fft.rfftfreq(n_fft, d=1/sr)
    min_bin = np.argmin(np.abs(freqs - 80)); max_bin = np.argmin(np.abs(freqs - 1000))
    bin_idx = np.argmax(mag[min_bin:max_bin]) + min_bin
    return float(freqs[bin_idx])

def estimate_pitch_acf(acf: np.ndarray, sr: int) -> float:
    min_lag = int(sr / 1000); max_lag = int(sr / 80)
    lag_range = acf[min_lag:max_lag]
    lag_idx = np.argmax(lag_range) + min_lag
    return float(sr / lag_idx)

def estimate_pitch_cepstrum(cep: np.ndarray, sr: int) -> float:
    min_lag = int(sr / 1000); max_lag = int(sr / 80)
    cep_range = np.abs(cep[min_lag:max_lag])
    q_idx = np.argmax(cep_range) + min_lag
    return float(sr / q_idx)

def estimate_pitch_cqt(cqt: np.ndarray, sr: int) -> float:
    fmin = librosa.note_to_hz('C1')
    freqs = librosa.cqt_frequencies(n_bins=len(cqt), fmin=fmin, bins_per_octave=12)
    bin_idx = np.argmax(cqt)
    return float(freqs[bin_idx])

def estimate_pitch_mel(mel: np.ndarray, sr: int) -> float:
    freqs = librosa.mel_frequencies(n_mels=len(mel), fmin=0.0, fmax=sr/2)
    bin_idx = np.argmax(mel)
    return float(freqs[bin_idx])

# ---------------------------------------------------------------------------
# Polynomial Ridge Regression (Pure NumPy)
# ---------------------------------------------------------------------------

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
    Solve closed-form Ridge Regression weights:
    W = inv(X^T X + alpha * I) X^T Y
    """
    # X_train shape: (N, D), Y_train shape: (N, Target_dim)
    D = X_train.shape[1]
    lhs = X_train.T @ X_train + alpha * np.eye(D)
    rhs = X_train.T @ Y_train
    return np.linalg.solve(lhs, rhs)

def evaluate_predictions(Y_true: np.ndarray, Y_pred: np.ndarray) -> tuple[float, float]:
    """
    Returns (Pearson correlation coefficient r, R^2 score).
    """
    # R^2 score
    ss_res = np.sum((Y_true - Y_pred)**2)
    ss_tot = np.sum((Y_true - np.mean(Y_true))**2)
    r2 = 1.0 - (ss_res / (ss_tot + 1e-10))
    
    # Pearson correlation coefficient r
    mean_true = np.mean(Y_true)
    mean_pred = np.mean(Y_pred)
    num = np.sum((Y_true - mean_true) * (Y_pred - mean_pred))
    denom = np.sqrt(np.sum((Y_true - mean_true)**2) * np.sum((Y_pred - mean_pred)**2)) + 1e-10
    r = num / denom
    
    return float(r), float(r2)

# ---------------------------------------------------------------------------
# High-energy Frame Extractor
# ---------------------------------------------------------------------------

def extract_high_energy_frames(y: np.ndarray, win: int = 1024, count: int = 400) -> list[np.ndarray]:
    frames = []
    # Dynamic hop size to cover the audio track evenly
    hop = max(1, len(y) // (count * 2))
    for i in range(0, len(y) - win, hop):
        frame = y[i:i+win]
        rms = np.sqrt(np.mean(frame**2))
        if rms > 0.02:
            # Normalize
            frame_norm = frame / (np.max(np.abs(frame)) + 1e-9)
            frames.append(frame_norm)
            if len(frames) >= count:
                break
    # Fill up with slight noise if file is too short/sparse
    while len(frames) < count:
        frames.append(np.random.normal(0, 0.01, win))
    return frames[:count]

# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------

def run():
    print("=" * 70)
    print("EXPERIMENT 028 — FAILURE MANIFOLD VALIDATION")
    print("=" * 70)
    sr = 22050
    win = 1024
    N_per_corpus = 400
    N_total = N_per_corpus * 5  # 2000 frames total

    print("\n[STRETCH TEST 2] Loading and synthesizing diverse signal corpora...")
    
    # 1. Vocals
    vocal_path = os.path.join(project_root, "Clean_vocal.wav")
    print(f"Loading real vocals from {vocal_path}...")
    y_voc, _ = librosa.load(vocal_path, sr=sr)
    vocals_frames = extract_high_energy_frames(y_voc, win, N_per_corpus)
    
    # 2. Speech
    print("Loading Speech example ('libri1')...")
    y_sp, _ = librosa.load(librosa.example('libri1'), sr=sr)
    speech_frames = extract_high_energy_frames(y_sp, win, N_per_corpus)
    
    # 3. Piano
    print("Loading Piano example ('pistachio')...")
    y_pn, _ = librosa.load(librosa.example('pistachio'), sr=sr)
    piano_frames = extract_high_energy_frames(y_pn, win, N_per_corpus)
    
    # 4. Drums
    print("Loading Drums example ('choice')...")
    y_dr, _ = librosa.load(librosa.example('choice'), sr=sr)
    drums_frames = extract_high_energy_frames(y_dr, win, N_per_corpus)
    
    # 5. Guitar
    print("Synthesizing plucked guitar notes (Karplus-Strong)...")
    guitar_frames = []
    rng = np.random.default_rng(42)
    f0s = [82.4, 110.0, 146.8, 196.0, 246.9, 329.6]
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

    print("Corpora sizes loaded:")
    for k, v in all_corpora.items():
        print(f" - {k:<8}: {len(v)} frames")
    
    print("\n[STRETCH TEST 3] Generating 2,000 perturbed frames across 12 perturbations...")
    
    # Gather reference representations for clean frames to compute similarity
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

    # Generate perturbed frames
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

    # Extract descriptors, compute similarities, and evaluate pitch/onset errors
    print("\n[STRETCH TEST 1] Extracting 17 failure descriptors & computing representation similarities...")
    X_features = []
    similarities = []
    pitch_errors = []
    onset_errors = []
    
    for i in range(N_total):
        frame_clean = all_corpora[corpus_labels[i]][i % N_per_corpus]
        frame_pert = perturbed_frames[i]
        
        # Representations for perturbed frame
        acf_p = compute_acf(frame_pert)
        cep_p = compute_cepstrum(frame_pert)
        mag_p = compute_stft(frame_pert, sr)
        if mag_p.ndim > 1: mag_p = np.mean(mag_p, axis=1)
        cqt_p = compute_cqt_frame(frame_pert, sr)
        cwt_p = compute_morlet_cwt(frame_pert, 64)
        mel_p = compute_mel_frame(frame_pert, sr)
        
        # Reference representations
        acf_c, cep_c, mag_c, cqt_c, cwt_c, mel_c = clean_representations[i]
        
        # Cosine similarities
        sim_stft = cosine_similarity(mag_c, mag_p)
        sim_acf  = cosine_similarity(acf_c, acf_p)
        sim_cep  = cosine_similarity(cep_c, cep_p)
        sim_cqt  = cosine_similarity(cqt_c, cqt_p)
        sim_cwt  = cosine_similarity(cwt_c, cwt_p)
        sim_mel  = cosine_similarity(mel_c, mel_p)
        
        similarities.append([sim_stft, sim_acf, sim_cep, sim_cqt, sim_cwt, sim_mel])
        
        # Descriptors
        feats = extract_17_descriptors(frame_pert, acf_p, cep_p, mag_p, cqt_p, cwt_p, mel_p, sr, 2 * (len(mag_p)-1))
        X_features.append(feats)
        
        # Pitch Error Evaluation (absolute semitone differences, capped at 12.0)
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

        p_mel_c = max(1.0, estimate_pitch_mel(mel_c, sr))
        p_mel_p = max(1.0, estimate_pitch_mel(mel_p, sr))
        e_mel = np.clip(12.0 * np.abs(np.log2(p_mel_p / p_mel_c)), 0, 12.0)
        
        # Average pitch estimation error across representations
        mean_pitch_err = np.mean([e_stft, e_acf, e_cep, e_cqt, e_mel])
        pitch_errors.append(mean_pitch_err)
        
        # Onset Error Evaluation: absolute difference in Spectral Flux (L1 norm of positive changes)
        flux_c = np.sum(np.maximum(0.0, mag_c))
        flux_p = np.sum(np.maximum(0.0, mag_p))
        mean_onset_err = np.abs(flux_c - flux_p)
        onset_errors.append(mean_onset_err)

    X = np.array(X_features)
    similarities = np.array(similarities)
    pitch_errors = np.array(pitch_errors)
    onset_errors = np.array(onset_errors)
    perturbation_labels = np.array(perturbation_labels)
    corpus_labels = np.array(corpus_labels)

    # ---------------------------------------------------------------------------
    # PCA Projections Comparison
    # ---------------------------------------------------------------------------
    print("\nRunning parallel PCA mappings to compare topologies...")
    
    # PCA-A: Original 11 features (STFT, ACF, Cepstrum)
    X_A, var_exp_A, _ = run_pca(X[:, :11])
    print(f"PCA-A (Original 11 descriptors) -> PC1: {var_exp_A[0]:.2%}, PC2: {var_exp_A[1]:.2%}, Total: {np.sum(var_exp_A):.2%}")
    
    # PCA-B: Expanded 17 features (STFT, ACF, Cepstrum + CQT, Wavelet, Mel)
    X_B, var_exp_B, _ = run_pca(X)
    print(f"PCA-B (Expanded 17 descriptors) -> PC1: {var_exp_B[0]:.2%}, PC2: {var_exp_B[1]:.2%}, Total: {np.sum(var_exp_B):.2%}")

    # ---------------------------------------------------------------------------
    # K-Means Clustering on Expanded PCA-B Space
    # ---------------------------------------------------------------------------
    print("\nRunning K-Means (k=5) on the expanded PCA-B manifold...")
    cluster_labels, centroids = run_kmeans(X_B, k=5, seed=42)
    
    cluster_profiles = {}
    print(f"{'Cluster':<8} {'Size':<5} {'STFT':<6} {'ACF':<6} {'Cep':<6} {'CQT':<6} {'CWT':<6} {'Mel':<6} {'Profile Label'}")
    print("-" * 88)
    for j in range(5):
        mask = (cluster_labels == j)
        size = np.sum(mask)
        if size == 0: continue
        m_sim = similarities[mask].mean(axis=0)
        
        # Labeling logic
        if m_sim[0] > 0.85 and m_sim[1] > 0.85 and m_sim[2] > 0.85:
            lbl = "Healthy / Low Degradation"
        elif m_sim[0] < 0.40 and m_sim[1] < 0.40 and m_sim[2] < 0.40:
            lbl = "Stochastic Noise Collapse"
        elif m_sim[2] < 0.35 and m_sim[1] > 0.65:
            lbl = "Periodicity Collapse (Cepstrum Drop)"
        elif m_sim[1] < 0.45 and m_sim[0] > 0.70:
            lbl = "Spectral Preserved / Periodicity Collapse"
        else:
            lbl = "Mixed Degradation boundary"
            
        cluster_profiles[j] = {"label": lbl, "sims": m_sim}
        print(f"C{j:<7} {size:<5} {m_sim[0]:<6.3f} {m_sim[1]:<6.3f} {m_sim[2]:<6.3f} {m_sim[3]:<6.3f} {m_sim[4]:<6.3f} {m_sim[5]:<6.3f} {lbl}")

    # ---------------------------------------------------------------------------
    # [STRETCH TEST 4] Predictive Power Validation
    # ---------------------------------------------------------------------------
    print("\n[STRETCH TEST 4] Validating predictive power of the manifold coordinates...")
    
    # Train/Test split (80% train, 20% test)
    N_train = int(0.8 * N_total)
    train_indices = rng.permutation(N_total)
    idx_train = train_indices[:N_train]
    idx_test = train_indices[N_train:]
    
    # Polynomial features for train/test based on PCA-B 2D coordinates
    Z_train = get_poly_features(X_B[idx_train])
    Z_test = get_poly_features(X_B[idx_test])
    
    # Targets for regression
    # 1. Pitch Error
    W_pitch = train_ridge_regression(Z_train, pitch_errors[idx_train], alpha=0.1)
    pitch_pred = Z_test @ W_pitch
    pitch_r, pitch_r2 = evaluate_predictions(pitch_errors[idx_test], pitch_pred)
    print(f"Pitch Error Prediction (PC1/PC2 only)    -> Pearson r: {pitch_r:.3f}, R^2: {pitch_r2:.3f}")
    
    # 2. Onset Error
    W_onset = train_ridge_regression(Z_train, onset_errors[idx_train], alpha=0.1)
    onset_pred = Z_test @ W_onset
    onset_r, onset_r2 = evaluate_predictions(onset_errors[idx_test], onset_pred)
    print(f"Onset Error Prediction (PC1/PC2 only)    -> Pearson r: {onset_r:.3f}, R^2: {onset_r2:.3f}")
    
    # 3. Fusion Routing (STFT, ACF, Cepstrum similarities)
    W_routing = train_ridge_regression(Z_train, similarities[idx_train, :3], alpha=0.1)
    routing_pred = Z_test @ W_routing
    
    # Measure mean R^2 for similarities prediction
    stft_r, stft_r2 = evaluate_predictions(similarities[idx_test, 0], routing_pred[:, 0])
    acf_r, acf_r2 = evaluate_predictions(similarities[idx_test, 1], routing_pred[:, 1])
    cep_r, cep_r2 = evaluate_predictions(similarities[idx_test, 2], routing_pred[:, 2])
    print(f"STFT Similarity Routing Prediction        -> Pearson r: {stft_r:.3f}, R^2: {stft_r2:.3f}")
    print(f"ACF Similarity Routing Prediction         -> Pearson r: {acf_r:.3f}, R^2: {acf_r2:.3f}")
    print(f"Cepstrum Similarity Routing Prediction    -> Pearson r: {cep_r:.3f}, R^2: {cep_r2:.3f}")

    # ---------------------------------------------------------------------------
    # Plotting
    # ---------------------------------------------------------------------------
    print("\nGenerating validation plots...")
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.25)
    
    # Panel 1: PCA-A vs PCA-B comparison
    ax_pca_a = fig.add_subplot(gs[0, 0])
    scatter_colors = ["#377eb8", "#e7298a", "#a6d854", "#ff7f00", "#984ea3"]
    # Plot PCA-A (original) colored by perturbation class
    unique_p = np.unique(perturbation_labels)
    for p_name in unique_p:
        mask = (perturbation_labels == p_name)
        ax_pca_a.scatter(X_A[mask, 0], X_A[mask, 1], s=8, alpha=0.6, label=p_name)
    ax_pca_a.set_title("PCA-A Manifold (Original 11 features)\nColored by Perturbation Type", fontsize=10, fontweight="bold")
    ax_pca_a.set_xlabel("PC1")
    ax_pca_a.set_ylabel("PC2")
    ax_pca_a.grid(True, alpha=0.08)
    
    ax_pca_b = fig.add_subplot(gs[0, 1])
    # Plot PCA-B (expanded) colored by K-Means clusters
    for j in range(5):
        mask = (cluster_labels == j)
        ax_pca_b.scatter(X_B[mask, 0], X_B[mask, 1], s=8, color=scatter_colors[j], alpha=0.7,
                         label=f"C{j}: {cluster_profiles[j]['label'][:25]}...")
    ax_pca_b.scatter(centroids[:, 0], centroids[:, 1], s=100, color="white", marker="X", edgecolors="black", label="Centroids")
    ax_pca_b.set_title("PCA-B Manifold (Expanded 17 features)\nColored by Discovered K-Means Clusters", fontsize=10, fontweight="bold")
    ax_pca_b.set_xlabel("PC1")
    ax_pca_b.set_ylabel("PC2")
    ax_pca_b.legend(fontsize=7, loc="lower left")
    ax_pca_b.grid(True, alpha=0.08)
    
    # Panel 3: PCA-B colored by Signal Corpora
    ax_corp = fig.add_subplot(gs[0, 2])
    unique_c = np.unique(corpus_labels)
    for c_name in unique_c:
        mask = (corpus_labels == c_name)
        ax_corp.scatter(X_B[mask, 0], X_B[mask, 1], s=8, alpha=0.6, label=c_name)
    ax_corp.set_title("PCA-B Manifold (Expanded 17 features)\nColored by Source Signal Category", fontsize=10, fontweight="bold")
    ax_corp.set_xlabel("PC1")
    ax_corp.set_ylabel("PC2")
    ax_corp.legend(fontsize=8, loc="upper right")
    ax_corp.grid(True, alpha=0.08)

    # Panel 4, 5, 6: CQT, Wavelet CWT, Mel similarities on PCA-B space
    ax_cqt = fig.add_subplot(gs[1, 0])
    sc_cqt = ax_cqt.scatter(X_B[:, 0], X_B[:, 1], s=6, c=similarities[:, 3], cmap="viridis", alpha=0.7)
    fig.colorbar(sc_cqt, ax=ax_cqt, label="CQT similarity")
    ax_cqt.set_title("CQT Similarity Map (Stress Test 1)", fontsize=10, fontweight="bold")
    ax_cqt.grid(True, alpha=0.08)
    
    ax_cwt = fig.add_subplot(gs[1, 1])
    sc_cwt = ax_cwt.scatter(X_B[:, 0], X_B[:, 1], s=6, c=similarities[:, 4], cmap="plasma", alpha=0.7)
    fig.colorbar(sc_cwt, ax=ax_cwt, label="Wavelet similarity")
    ax_cwt.set_title("Wavelet CWT Similarity Map (Stress Test 1)", fontsize=10, fontweight="bold")
    ax_cwt.grid(True, alpha=0.08)

    ax_mel = fig.add_subplot(gs[1, 2])
    sc_mel = ax_mel.scatter(X_B[:, 0], X_B[:, 1], s=6, c=similarities[:, 5], cmap="magma", alpha=0.7)
    fig.colorbar(sc_mel, ax=ax_mel, label="Mel similarity")
    ax_mel.set_title("Mel Spectrogram Similarity Map (Stress Test 1)", fontsize=10, fontweight="bold")
    ax_mel.grid(True, alpha=0.08)

    # Panel 7: Pitch tracker error predictions on Test Set
    ax_pred_pitch = fig.add_subplot(gs[2, 0])
    ax_pred_pitch.scatter(pitch_errors[idx_test], pitch_pred, s=15, color="#e7298a", alpha=0.5)
    # Reference line
    lims = [0, 12]
    ax_pred_pitch.plot(lims, lims, 'w--', alpha=0.5, label="Perfect correlation")
    ax_pred_pitch.set_title(f"Pitch Error Prediction (Test Split)\nPearson r: {pitch_r:.3f}, R^2: {pitch_r2:.3f}", fontsize=10, fontweight="bold")
    ax_pred_pitch.set_xlabel("Actual Pitch Error (semitones)")
    ax_pred_pitch.set_ylabel("Predicted Pitch Error (semitones)")
    ax_pred_pitch.legend(fontsize=8)
    ax_pred_pitch.grid(True, alpha=0.08)

    # Panel 8: Onset tracker error predictions on Test Set
    ax_pred_onset = fig.add_subplot(gs[2, 1])
    ax_pred_onset.scatter(onset_errors[idx_test], onset_pred, s=15, color="#a6d854", alpha=0.5)
    ax_pred_onset.set_title(f"Onset Error Prediction (Test Split)\nPearson r: {onset_r:.3f}, R^2: {onset_r2:.3f}", fontsize=10, fontweight="bold")
    ax_pred_onset.set_xlabel("Actual Onset Error (Spectral Flux delta)")
    ax_pred_onset.set_ylabel("Predicted Onset Error (Spectral Flux delta)")
    ax_pred_onset.grid(True, alpha=0.08)

    # Panel 9: Cluster similarities bar chart (original vs new)
    ax_bar = fig.add_subplot(gs[2, 2])
    x = np.arange(5)
    width = 0.15
    
    stft_means = [cluster_profiles[j]["sims"][0] for j in range(5)]
    acf_means  = [cluster_profiles[j]["sims"][1] for j in range(5)]
    cep_means  = [cluster_profiles[j]["sims"][2] for j in range(5)]
    cqt_means  = [cluster_profiles[j]["sims"][3] for j in range(5)]
    cwt_means  = [cluster_profiles[j]["sims"][4] for j in range(5)]
    mel_means  = [cluster_profiles[j]["sims"][5] for j in range(5)]
    
    ax_bar.bar(x - 2*width, stft_means, width, label="STFT", color="#a6d854", alpha=0.8)
    ax_bar.bar(x - width, acf_means, width, label="ACF", color="#377eb8", alpha=0.8)
    ax_bar.bar(x, cep_means, width, label="Cepstrum", color="#e7298a", alpha=0.8)
    ax_bar.bar(x + width, cqt_means, width, label="CQT", color="#ff7f00", alpha=0.8)
    ax_bar.bar(x + 2*width, cwt_means, width, label="Wavelet", color="#984ea3", alpha=0.8)
    
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([f"C{j}" for j in range(5)])
    ax_bar.set_ylabel("Mean Cosine Similarity")
    ax_bar.set_title("Mean Representation Similarity\nacross Discovered K-Means Clusters", fontsize=10, fontweight="bold")
    ax_bar.legend(fontsize=7, loc="upper right")
    ax_bar.grid(True, alpha=0.08)

    fig.suptitle("Experiment 028 — Failure Manifold Validation:\nStress-Testing Manifold Topology & Predictive Power across 5 Corpora & 12 Perturbations", fontsize=14, fontweight="bold", y=0.98)
    
    out_path = os.path.join(project_root, "results", "exp028_failure_manifold_validation.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\nSaved validation plot: {out_path}")
    print("=" * 70)


if __name__ == "__main__":
    run()
