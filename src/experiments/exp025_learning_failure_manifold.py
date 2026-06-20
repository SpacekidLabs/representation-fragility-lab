"""
Experiment 025 — Learning the Failure Manifold
==============================================
Replacing handcrafted heuristics with a learned model predicting estimation error
from a multi-dimensional feature space of failure descriptors.

Key Steps:
1. Synthesise training and testing pitch sweeps.
2. Extract 12 failure descriptors per frame.
3. Measure absolute pitch error in semitones (capped at 12.0).
4. Train Ridge Regression models to predict expected errors:
   Expect_Error = X * W
5. Compute dynamic fusion weights:
   weight = 1 / (Expect_Error + 0.05)
6. Evaluate tracking accuracy vs. baselines under multiple perturbations.
7. Generate plot results/exp025_learning_failure_manifold.png.
"""

import sys
import os
import numpy as np
import scipy.signal
import scipy.io.wavfile
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
# Signal Synthesis and Perturbations
# ---------------------------------------------------------------------------

def synthesise_sweep(sr: int, duration: float, f_start: float, f_end: float,
                     vibrato: bool = False, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """
    Synthesise a harmonic frequency sweep. Returns (audio, true_pitch_per_sample).
    """
    rng = np.random.default_rng(seed)
    n_samples = int(duration * sr)
    t = np.arange(n_samples) / sr

    if vibrato:
        # Sinusoidal pitch modulation (vibrato)
        # instantaneous frequency = f_start + sweep + vibrato
        f_drift = f_start + (f_end - f_start) * (t / duration)
        freqs_inst = f_drift + 40.0 * np.sin(2 * np.pi * 5.0 * t)
        # phase is the integral of instantaneous frequency
        phase = 2 * np.pi * (f_start * t + 0.5 * (f_end - f_start) / duration * t**2 - 
                             (40.0 / (2 * np.pi * 5.0)) * np.cos(2 * np.pi * 5.0 * t))
    else:
        # Linear frequency sweep
        freqs_inst = f_start + (f_end - f_start) * (t / duration)
        phase = 2 * np.pi * (f_start * t + 0.5 * (f_end - f_start) / duration * t**2)

    # Generate 5-harmonic stack
    audio = np.zeros(n_samples)
    for k in range(1, 6):
        audio += (1.0 / k) * np.sin(k * phase)

    # Normalise and apply soft fade-in/out envelope to prevent clicks
    env = np.ones(n_samples)
    fade = int(0.02 * sr)
    env[:fade] = np.linspace(0.0, 1.0, fade)
    env[-fade:] = np.linspace(1.0, 0.0, fade)
    audio *= env

    audio /= np.max(np.abs(audio)) + 1e-9
    return audio, freqs_inst


def lowpass_filter(audio: np.ndarray, sr: int, cutoff: float = 600.0) -> np.ndarray:
    b, a = scipy.signal.butter(4, cutoff / (sr / 2.0), btype='low')
    return scipy.signal.filtfilt(b, a, audio)


def hard_clip(audio: np.ndarray, threshold: float = 0.15) -> np.ndarray:
    clipped = np.clip(audio, -threshold, threshold)
    return clipped / (np.max(np.abs(clipped)) + 1e-9)


# ---------------------------------------------------------------------------
# Pitch Estimators
# ---------------------------------------------------------------------------

def estimate_pitch_acf(acf: np.ndarray, sr: int) -> tuple[float, int]:
    min_lag = int(sr / 1000)
    max_lag = int(sr / 80)
    if max_lag > len(acf):
        max_lag = len(acf)
    acf_range = acf[min_lag:max_lag]
    lag_idx = np.argmax(acf_range) + min_lag
    pitch = sr / lag_idx
    return float(pitch), int(lag_idx)


def estimate_pitch_cepstrum(cep: np.ndarray, sr: int) -> tuple[float, int]:
    min_lag = int(sr / 1000)
    max_lag = int(sr / 80)
    if max_lag > len(cep):
        max_lag = len(cep)
    cep_range = np.abs(cep[min_lag:max_lag])
    q_idx = np.argmax(cep_range) + min_lag
    pitch = sr / q_idx
    return float(pitch), int(q_idx)


def estimate_pitch_stft(mag: np.ndarray, sr: int, n_fft: int = 2048) -> tuple[float, int]:
    freqs = np.fft.rfftfreq(n_fft, d=1/sr)
    min_bin = np.argmin(np.abs(freqs - 80))
    max_bin = np.argmin(np.abs(freqs - 1000))
    spec_range = mag[min_bin:max_bin]
    bin_idx = np.argmax(spec_range) + min_bin
    pitch = freqs[bin_idx]
    return float(pitch), int(bin_idx)


# ---------------------------------------------------------------------------
# Handcrafted Confidence Heuristics (from Exp 015/022)
# ---------------------------------------------------------------------------

def get_handcrafted_confidence(acf: np.ndarray, cep: np.ndarray, mag: np.ndarray,
                               sr: int, win: int) -> tuple[float, float, float]:
    # ACF prominence ratio
    min_lag = int(sr / 1000)
    max_lag = int(sr / 80)
    acf_peak = np.max(acf[min_lag:max_lag])
    acf_ratio = acf_peak / (acf[0] + 1e-10)
    acf_conf = float(np.clip((acf_ratio - 0.15) / (0.75 - 0.15), 0.0, 1.0))

    # Cepstrum inverse DC shift
    cep_c0 = cep[0]
    cep_conf = float(np.clip(1.0 - (cep_c0 - (-23.0)) / 25.4, 0.0, 1.0))

    # STFT peak dominance ratio
    freqs = np.fft.rfftfreq(2 * (len(mag) - 1), d=1/sr) if len(mag) > 1 else np.array([1])
    min_bin = np.argmin(np.abs(freqs - 80))
    max_bin = np.argmin(np.abs(freqs - 1000))
    spec_range = mag[min_bin:max_bin]
    stft_peak = np.max(spec_range)
    stft_mean = np.mean(spec_range)
    stft_conf = float(np.clip((stft_peak - stft_mean) / (stft_peak + 1e-10), 0.0, 1.0))

    return acf_conf, cep_conf, stft_conf


# ---------------------------------------------------------------------------
# Feature Extraction (11 static + 1 dynamic = 12 total)
# ---------------------------------------------------------------------------

def extract_descriptors(frame: np.ndarray, acf: np.ndarray, cep: np.ndarray,
                        mag: np.ndarray, sr: int, n_fft: int = 2048) -> list[float]:
    min_lag = int(sr / 1000)
    max_lag = int(sr / 80)
    freqs = np.fft.rfftfreq(n_fft, d=1/sr)
    min_bin = np.argmin(np.abs(freqs - 80))
    max_bin = np.argmin(np.abs(freqs - 1000))
    min_q = min_lag
    max_q = max_lag

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
        geom_mean = np.exp(log_mean)
        arith_mean = np.mean(spec_range)
        spec_flatness = geom_mean / (arith_mean + 1e-12)
    else:
        spec_flatness = 1.0

    stft_p, stft_peak_idx = estimate_pitch_stft(mag, sr, n_fft)

    # 3. STFT Peak Strength
    if spec_sum > 1e-9:
        stft_peak_strength = mag[stft_peak_idx] / spec_sum
    else:
        stft_peak_strength = 0.0

    # 4. STFT Peak Prominence
    stft_peak_prominence = mag[stft_peak_idx] - np.mean(spec_range)
    if mag[stft_peak_idx] > 0:
        stft_peak_prominence /= (mag[stft_peak_idx] + 1e-12)
    stft_peak_prominence = max(0.0, float(stft_peak_prominence))

    acf_p, acf_peak_idx = estimate_pitch_acf(acf, sr)

    # 5. ACF Peak Strength
    if acf[0] > 1e-9:
        acf_peak_strength = acf[acf_peak_idx] / acf[0]
    else:
        acf_peak_strength = 0.0

    # 6. ACF Peak Prominence
    acf_range = acf[min_lag:max_lag]
    acf_mean = np.mean(acf_range)
    acf_max = np.max(acf_range)
    acf_min = np.min(acf_range)
    acf_peak_prominence = (acf[acf_peak_idx] - acf_mean) / (acf_max - acf_min + 1e-10)
    acf_peak_prominence = float(np.clip(acf_peak_prominence, 0.0, 1.0))

    # 7. Cepstrum DC Coefficient c0
    cep_c0 = float(cep[0])

    cep_p, cep_peak_idx = estimate_pitch_cepstrum(cep, sr)

    # 8. Cepstral Peak Strength
    cep_peak_strength = float(np.abs(cep[cep_peak_idx]) / (np.abs(cep_c0) + 1e-10))

    # 9. Cepstral Peak Prominence
    cep_range = np.abs(cep[min_q:max_q])
    cep_mean = np.mean(cep_range)
    cep_max = np.max(cep_range)
    cep_min = np.min(cep_range)
    cep_peak_prominence = (np.abs(cep[cep_peak_idx]) - cep_mean) / (cep_max - cep_min + 1e-10)
    cep_peak_prominence = float(np.clip(cep_peak_prominence, 0.0, 1.0))

    # 10. Zero Crossing Rate (ZCR)
    zcr = float(np.mean(np.abs(np.diff(np.sign(frame)))) / 2.0)

    # 11. Frame Log Energy
    energy = float(np.mean(frame**2))
    frame_log_energy = float(np.log(energy + 1e-10))

    return [
        spec_entropy,
        spec_flatness,
        stft_peak_strength,
        stft_peak_prominence,
        acf_peak_strength,
        acf_peak_prominence,
        cep_c0,
        cep_peak_strength,
        cep_peak_prominence,
        zcr,
        frame_log_energy,
        float(acf_peak_idx) # return peak lag to compute jitter dynamically
    ]


# ---------------------------------------------------------------------------
# Dataset Generation and Feature Gathering
# ---------------------------------------------------------------------------

def collect_data_from_signal(audio: np.ndarray, true_pitch_per_sample: np.ndarray,
                             sr: int, win: int, hop: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Process signal block-by-block. 
    Returns:
      X: feature matrix of shape (N_frames, 13) (12 features + bias)
      Y_err: pitch errors of shape (N_frames, 3) (STFT, ACF, Cepstrum)
      P_est: estimated pitches of shape (N_frames, 3)
    """
    features = []
    errors = []
    pitches = []

    num_frames = (len(audio) - win) // hop + 1
    acf_lags_history = []

    for n in range(num_frames):
        start = n * hop
        frame_raw = audio[start:start + win]
        frame_windowed = frame_raw * np.hanning(win)

        acf = compute_acf(frame_windowed)
        cep = compute_cepstrum(frame_windowed)
        mag = compute_stft(frame_windowed, sr)
        if mag.ndim > 1:
            mag = np.mean(mag, axis=1)

        # Get pitch estimates
        p_stft, _ = estimate_pitch_stft(mag, sr, 2 * (len(mag) - 1))
        p_acf, acf_peak_idx = estimate_pitch_acf(acf, sr)
        p_cep, _ = estimate_pitch_cepstrum(cep, sr)

        # Track ACF lag index for jitter computation
        acf_lags_history.append(float(acf_peak_idx))
        if len(acf_lags_history) > 3:
            acf_lags_history.pop(0)
        acf_jitter = float(np.var(acf_lags_history))

        # Extract features
        feats = extract_descriptors(frame_raw, acf, cep, mag, sr, 2 * (len(mag) - 1))
        # Replace the peak lag element with jitter
        feats[-1] = acf_jitter
        # Add bias
        feats.append(1.0)

        # Compute ground truth pitch at the centre of the frame
        frame_center_sample = start + win // 2
        p_true = true_pitch_per_sample[min(frame_center_sample, len(true_pitch_per_sample)-1)]

        # Compute error in semitones capped at 12.0
        def semitone_err(p_est):
            if p_est <= 0:
                return 12.0
            return min(12.0 * np.abs(np.log2(p_est / p_true)), 12.0)

        errs = [semitone_err(p_stft), semitone_err(p_acf), semitone_err(p_cep)]

        features.append(feats)
        errors.append(errs)
        pitches.append([p_stft, p_acf, p_cep])

    return np.array(features), np.array(errors), np.array(pitches)


def generate_training_dataset(sr: int, win: int, hop: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Synthesise 6 sweep conditions and compile into matrices X and Y.
    """
    X_list = []
    Y_list = []

    # Condition 1: Clean Sweep
    audio, true_p = synthesise_sweep(sr, duration=2.5, f_start=180.0, f_end=550.0, seed=10)
    X, Y, _ = collect_data_from_signal(audio, true_p, sr, win, hop)
    X_list.append(X); Y_list.append(Y)

    # Condition 2: Noisy Sweep
    audio, true_p = synthesise_sweep(sr, duration=2.5, f_start=180.0, f_end=550.0, seed=11)
    audio += np.random.normal(0.0, 0.15, len(audio))
    X, Y, _ = collect_data_from_signal(audio, true_p, sr, win, hop)
    X_list.append(X); Y_list.append(Y)

    # Condition 3: Filtered Sweep
    audio, true_p = synthesise_sweep(sr, duration=2.5, f_start=180.0, f_end=550.0, seed=12)
    audio = lowpass_filter(audio, sr, cutoff=400.0)
    X, Y, _ = collect_data_from_signal(audio, true_p, sr, win, hop)
    X_list.append(X); Y_list.append(Y)

    # Condition 4: Distorted Sweep
    audio, true_p = synthesise_sweep(sr, duration=2.5, f_start=180.0, f_end=550.0, vibrato=True, seed=13)
    audio = hard_clip(audio, threshold=0.08)
    X, Y, _ = collect_data_from_signal(audio, true_p, sr, win, hop)
    X_list.append(X); Y_list.append(Y)

    # Condition 5: Vibrato Sweep (Clean)
    audio, true_p = synthesise_sweep(sr, duration=2.5, f_start=180.0, f_end=550.0, vibrato=True, seed=14)
    X, Y, _ = collect_data_from_signal(audio, true_p, sr, win, hop)
    X_list.append(X); Y_list.append(Y)

    # Condition 6: Vibrato Sweep (Noisy)
    audio, true_p = synthesise_sweep(sr, duration=2.5, f_start=180.0, f_end=550.0, vibrato=True, seed=15)
    audio += np.random.normal(0.0, 0.10, len(audio))
    X, Y, _ = collect_data_from_signal(audio, true_p, sr, win, hop)
    X_list.append(X); Y_list.append(Y)

    return np.concatenate(X_list, axis=0), np.concatenate(Y_list, axis=0)


# ---------------------------------------------------------------------------
# Training Model (Ridge Regression)
# ---------------------------------------------------------------------------

def train_failure_models(X: np.ndarray, Y: np.ndarray, alpha: float = 0.05) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """
    Fits Ridge Regression weights for each representation after standardising features.
    """
    # Standardise all columns except the bias term (last column)
    X_feats = X[:, :-1]
    mu = np.mean(X_feats, axis=0)
    sigma = np.std(X_feats, axis=0) + 1e-8
    
    X_std = X.copy()
    X_std[:, :-1] = (X_feats - mu) / sigma

    weights = []
    N_features = X_std.shape[1]
    I = np.eye(N_features)
    # Don't regularise the bias term (last column)
    I[-1, -1] = 0.0

    for idx in range(3):
        y = Y[:, idx]
        # Direct closed-form normal equation solve
        w = np.linalg.pinv(X_std.T @ X_std + alpha * I) @ X_std.T @ y
        weights.append(w)
    return weights, mu, sigma


# ---------------------------------------------------------------------------
# Evaluation Routine
# ---------------------------------------------------------------------------

def evaluate_models(audio: np.ndarray, true_pitch: np.ndarray, W_models: list[np.ndarray],
                    mu: np.ndarray, sigma: np.ndarray, sr: int, win: int, hop: int) -> dict:
    """
    Fuses pitch and logs results for the test signal.
    """
    X, Y_true, P_est = collect_data_from_signal(audio, true_pitch, sr, win, hop)
    t = np.array([(n * hop + win // 2) / sr for n in range(len(X))])

    # Standardise test features using training statistics
    X_std = X.copy()
    X_std[:, :-1] = (X[:, :-1] - mu) / sigma

    # True error vectors
    err_stft_true = Y_true[:, 0]
    err_acf_true  = Y_true[:, 1]
    err_cep_true  = Y_true[:, 2]

    # Predict expected error using standardised features
    err_stft_pred = np.maximum(X_std @ W_models[0], 0.0)
    err_acf_pred  = np.maximum(X_std @ W_models[1], 0.0)
    err_cep_pred  = np.maximum(X_std @ W_models[2], 0.0)

    # Dynamic Weighting based on predicted error
    # raw_weight = 1.0 / (predicted_error_semitones + smoothing)
    eps = 0.05
    w_stft = 1.0 / (err_stft_pred + eps)
    w_acf  = 1.0 / (err_acf_pred  + eps)
    w_cep  = 1.0 / (err_cep_pred  + eps)
    w_sum  = w_stft + w_acf + w_cep + 1e-10

    w_stft /= w_sum
    w_acf  /= w_sum
    w_cep  /= w_sum

    # Fused pitch
    fused_pitch_manifold = w_stft * P_est[:, 0] + w_acf * P_est[:, 1] + w_cep * P_est[:, 2]

    # Reactive benchmark weights
    reactive_weights = []
    for i in range(len(X)):
        # Extract representations again to query reactive confidence
        start = i * hop
        frame = audio[start:start + win] * np.hanning(win)
        acf = compute_acf(frame)
        cep = compute_cepstrum(frame)
        mag = compute_stft(frame, sr)
        if mag.ndim > 1:
            mag = np.mean(mag, axis=1)
        r_acf, r_cep, r_stft = get_handcrafted_confidence(acf, cep, mag, sr, win)
        reactive_weights.append([r_stft, r_acf, r_cep])
    
    rw = np.array(reactive_weights)
    rw_sum = rw.sum(axis=1, keepdims=True) + 1e-10
    rw /= rw_sum
    fused_pitch_reactive = rw[:, 0] * P_est[:, 0] + rw[:, 1] * P_est[:, 1] + rw[:, 2] * P_est[:, 2]

    # R2 scores/correlations
    def correlation(true, pred):
        if np.std(true) < 1e-9 or np.std(pred) < 1e-9:
            return 0.0
        return float(np.corrcoef(true, pred)[0, 1])

    corr_stft = correlation(err_stft_true, err_stft_pred)
    corr_acf  = correlation(err_acf_true, err_acf_pred)
    corr_cep  = correlation(err_cep_true, err_cep_pred)

    results = {
        "t": t,
        "true_pitch": np.array([true_pitch[min(int(s * sr + win // 2), len(true_pitch)-1)] for s in t]),
        "pitches": P_est,
        "fused_manifold": fused_pitch_manifold,
        "fused_reactive": fused_pitch_reactive,
        "true_errors": Y_true,
        "pred_errors": np.column_stack([err_stft_pred, err_acf_pred, err_cep_pred]),
        "weights": np.column_stack([w_stft, w_acf, w_cep]),
        "correlations": [corr_stft, corr_acf, corr_cep]
    }
    return results


# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------

def run():
    print("=" * 60)
    print("EXPERIMENT 025 — LEARNING THE FAILURE MANIFOLD")
    print("=" * 60)

    sr = 22050
    WIN = 1024   # 46 ms
    HOP = 256    # 11.6 ms

    print("Generating training dataset (sweeps under noise, filtering, clipping)...")
    X_train, Y_train = generate_training_dataset(sr, WIN, HOP)
    print(f"Data gathered. Training matrix X: {X_train.shape}, Labels Y: {Y_train.shape}")

    print("Training Ridge models to predict error from 12 failure descriptors...")
    W_models, mu, sigma = train_failure_models(X_train, Y_train, alpha=0.1)
    print("Training complete.\n")

    # Generate Test Signal: Vibrato Sweep (300Hz center, sweeping, 5Hz vibrato)
    print("Generating vibrato test signal...")
    audio_clean, true_pitch = synthesise_sweep(sr, duration=3.0, f_start=250.0, f_end=450.0, vibrato=True, seed=99)

    test_conditions = {
        "Clean": audio_clean,
        "Noisy (σ=0.20)": audio_clean + np.random.normal(0.0, 0.20, len(audio_clean)),
        "Filtered (LP=400Hz)": lowpass_filter(audio_clean, sr, cutoff=400.0),
        "Distorted (clip=0.10)": hard_clip(audio_clean, threshold=0.10)
    }

    # Evaluate across conditions
    all_results = {}
    print(f"{'Condition':<22}  {'Estimator':<12}  {'Mean Abs Error (semitones)':>28}")
    print("-" * 66)

    for cond_name, audio in test_conditions.items():
        res = evaluate_models(audio, true_pitch, W_models, mu, sigma, sr, WIN, HOP)
        all_results[cond_name] = res

        # Extract pitch vectors
        p_true = res["true_pitch"]
        p_fused_man = res["fused_manifold"]
        p_fused_re  = res["fused_reactive"]
        p_stft = res["pitches"][:, 0]
        p_acf  = res["pitches"][:, 1]
        p_cep  = res["pitches"][:, 2]

        def mae(p_est):
            errs = 12.0 * np.abs(np.log2(p_est / p_true))
            return np.mean(errs)

        print(f"{cond_name:<22}  {'STFT-only':<12}  {mae(p_stft):>28.3f}")
        print(f"{cond_name:<22}  {'ACF-only':<12}  {mae(p_acf):>28.3f}")
        print(f"{cond_name:<22}  {'Cep-only':<12}  {mae(p_cep):>28.3f}")
        print(f"{cond_name:<22}  {'Reactive Hybrid':<12}  {mae(p_fused_re):>28.3f}")
        print(f"{cond_name:<22}  {'Manifold Learned':<12}  {mae(p_fused_man):>28.3f} ◀")
        print(f"Correlation (True vs Pred Error) -> STFT: {res['correlations'][0]:.2f} | ACF: {res['correlations'][1]:.2f} | Cepstrum: {res['correlations'][2]:.2f}")
        print()

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    print("Generating summary and diagnostics plot...")
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(16, 14))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.25)

    # Let's plot the "Noisy" condition details as the representative diagnostic
    noisy_res = all_results["Noisy (σ=0.20)"]
    t = noisy_res["t"]
    
    # 1. STFT True vs Pred Error (Noisy Condition)
    ax_stft = fig.add_subplot(gs[0, 0])
    ax_stft.plot(t, noisy_res["true_errors"][:, 0], color="#a6d854", alpha=0.4, label="True STFT Error")
    ax_stft.plot(t, noisy_res["pred_errors"][:, 0], color="#a6d854", lw=1.8, label="Predicted STFT Error")
    ax_stft.set_title(f"STFT: True vs Predicted Error (Corr: {noisy_res['correlations'][0]:.2f})", fontsize=10, fontweight="bold")
    ax_stft.set_ylabel("Error (semitones)")
    ax_stft.legend(fontsize=8)
    ax_stft.grid(True, alpha=0.1)

    # 2. ACF True vs Pred Error (Noisy Condition)
    ax_acf = fig.add_subplot(gs[0, 1])
    ax_acf.plot(t, noisy_res["true_errors"][:, 1], color="#377eb8", alpha=0.4, label="True ACF Error")
    ax_acf.plot(t, noisy_res["pred_errors"][:, 1], color="#377eb8", lw=1.8, label="Predicted ACF Error")
    ax_acf.set_title(f"ACF: True vs Predicted Error (Corr: {noisy_res['correlations'][1]:.2f})", fontsize=10, fontweight="bold")
    ax_acf.legend(fontsize=8)
    ax_acf.grid(True, alpha=0.1)

    # 3. Cepstrum True vs Pred Error (Noisy Condition)
    ax_cep = fig.add_subplot(gs[1, 0])
    ax_cep.plot(t, noisy_res["true_errors"][:, 2], color="#e7298a", alpha=0.4, label="True Cep Error")
    ax_cep.plot(t, noisy_res["pred_errors"][:, 2], color="#e7298a", lw=1.8, label="Predicted Cep Error")
    ax_cep.set_title(f"Cepstrum: True vs Predicted Error (Corr: {noisy_res['correlations'][2]:.2f})", fontsize=10, fontweight="bold")
    ax_cep.set_ylabel("Error (semitones)")
    ax_cep.legend(fontsize=8)
    ax_cep.grid(True, alpha=0.1)

    # 4. Learned Fusion weights over time (Noisy Condition)
    ax_w = fig.add_subplot(gs[1, 1])
    ax_w.stackplot(t, noisy_res["weights"][:, 0], noisy_res["weights"][:, 1], noisy_res["weights"][:, 2],
                   labels=["STFT Weight", "ACF Weight", "Cep Weight"],
                   colors=["#a6d854", "#377eb8", "#e7298a"], alpha=0.7)
    ax_w.set_title("Learned Failure Manifold Dynamic Weights (Noisy Condition)", fontsize=10, fontweight="bold")
    ax_w.set_ylim(0, 1.0)
    ax_w.legend(fontsize=8, loc="lower left")
    ax_w.grid(True, alpha=0.1)

    # 5. Fused Pitch Tracking comparison (Noisy Condition)
    ax_pitch = fig.add_subplot(gs[2, 0])
    ax_pitch.plot(t, noisy_res["true_pitch"], color="white", lw=2.2, label="True Pitch (Vibrato Sweep)")
    ax_pitch.plot(t, noisy_res["fused_reactive"], color="#ff7f00", lw=1.2, alpha=0.8, label="Reactive Hybrid")
    ax_pitch.plot(t, noisy_res["fused_manifold"], color="#33cc66", lw=1.8, label="Manifold Learned Hybrid")
    ax_pitch.set_title("Pitch Tracking Comparison (Noisy Condition)", fontsize=10, fontweight="bold")
    ax_pitch.set_ylabel("Frequency (Hz)")
    ax_pitch.set_xlabel("Time (seconds)")
    ax_pitch.legend(fontsize=8)
    ax_pitch.grid(True, alpha=0.1)

    # 6. Overall Summary comparison bar chart
    ax_bar = fig.add_subplot(gs[2, 1])
    methods = ["STFT", "ACF", "Cepstrum", "Reactive", "Manifold Learned"]
    colors = ["#a6d854", "#377eb8", "#e7298a", "#ff7f00", "#33cc66"]
    
    x = np.arange(len(test_conditions))
    width = 0.15
    offsets = np.linspace(-2, 2, 5) * width

    for m_idx, method in enumerate(methods):
        maes = []
        for cond_name, res in all_results.items():
            p_true = res["true_pitch"]
            if method == "STFT":
                p_est = res["pitches"][:, 0]
            elif method == "ACF":
                p_est = res["pitches"][:, 1]
            elif method == "Cepstrum":
                p_est = res["pitches"][:, 2]
            elif method == "Reactive":
                p_est = res["fused_reactive"]
            else:
                p_est = res["fused_manifold"]
            
            errs = 12.0 * np.abs(np.log2(p_est / p_true))
            maes.append(np.mean(errs))

        bars = ax_bar.bar(x + offsets[m_idx], maes, width * 0.9,
                          label=method, color=colors[m_idx], alpha=0.85)
        
        # Add values on top of bars
        for bar in bars:
            height = bar.get_height()
            ax_bar.text(bar.get_x() + bar.get_width() / 2, height + 0.1,
                        f"{height:.1f}", ha="center", va="bottom", fontsize=7, color="white")

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(list(test_conditions.keys()), fontsize=9)
    ax_bar.set_ylabel("Mean Absolute Error (semitones)")
    ax_bar.set_title("Overall Pitch Tracking Error Comparison", fontsize=10, fontweight="bold")
    ax_bar.legend(fontsize=8)
    ax_bar.grid(True, alpha=0.1, axis="y")

    fig.suptitle("Experiment 025 — Learning the Failure Manifold", fontsize=14, fontweight="bold", y=0.98)
    out_path = os.path.join(project_root, "results", "exp025_learning_failure_manifold.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    run()
