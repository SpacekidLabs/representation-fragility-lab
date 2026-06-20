"""
Experiment 026 — Cross-Task Transfer
====================================
Evaluating if the failure models trained on pitch tracking can generalise
to route fusion weights for onset detection, proving the universality
of representation failure geometry.

Architecture:
1. Train failure models on pitch tracking (Experiment 025).
2. Generate multi-note sequences for onset detection (Experiment 024) under:
   - Clean
   - Noisy (sigma=0.15)
   - Filtered (LP=400Hz)
   - Distorted (clip=0.08)
3. Predict expected pitch errors on onset frames.
4. Route onset fusion weights inversely proportional to predicted pitch error:
   weight = 1 / (Predicted_Pitch_Error + 0.05)
5. Compare F1 scores against STFT, ACF, Cepstrum, Reactive, and Onset-Trained Meta models.
6. Generate plot results/exp026_cross_task_transfer.png.
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
# Signal Synthesis (Pitch Sweeps - Exp 025)
# ---------------------------------------------------------------------------

def synthesise_sweep(sr: int, duration: float, f_start: float, f_end: float,
                     vibrato: bool = False, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    n_samples = int(duration * sr)
    t = np.arange(n_samples) / sr

    if vibrato:
        f_drift = f_start + (f_end - f_start) * (t / duration)
        freqs_inst = f_drift + 40.0 * np.sin(2 * np.pi * 5.0 * t)
        phase = 2 * np.pi * (f_start * t + 0.5 * (f_end - f_start) / duration * t**2 - 
                             (40.0 / (2 * np.pi * 5.0)) * np.cos(2 * np.pi * 5.0 * t))
    else:
        freqs_inst = f_start + (f_end - f_start) * (t / duration)
        phase = 2 * np.pi * (f_start * t + 0.5 * (f_end - f_start) / duration * t**2)

    audio = np.zeros(n_samples)
    for k in range(1, 6):
        audio += (1.0 / k) * np.sin(k * phase)

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
# Onset Audio Synthesis (Multi-note sequence - Exp 024)
# ---------------------------------------------------------------------------

def synthesise_onset_sequence(sr: int, seed: int = 0) -> tuple[np.ndarray, list[float]]:
    rng = np.random.default_rng(seed)
    notes = [
        (220.0, 0.30),
        (330.0, 0.25),
        (440.0, 0.35),
        (330.0, 0.20),
        (550.0, 0.30),
        (220.0, 0.25),
        (440.0, 0.30),
        (330.0, 0.20),
    ]
    gap = 0.04
    audio_chunks = []
    onset_times  = []
    cursor = 0.0

    for freq, dur in notes:
        gap_samples = int(gap * sr)
        audio_chunks.append(np.zeros(gap_samples))
        cursor += gap
        onset_times.append(cursor)

        n_samples = int(dur * sr)
        t = np.arange(n_samples) / sr
        note = np.zeros(n_samples)
        for k in range(1, 6):
            note += (1.0 / k) * np.sin(2 * np.pi * k * freq * t)

        env = np.ones(n_samples)
        atk = min(int(0.005 * sr), n_samples // 4)
        rel = min(int(0.020 * sr), n_samples // 4)
        env[:atk] = np.linspace(0.0, 1.0, atk)
        env[-rel:] = np.linspace(1.0, 0.0, rel)
        note *= env

        audio_chunks.append(note)
        cursor += dur

    audio = np.concatenate(audio_chunks)
    audio /= np.max(np.abs(audio)) + 1e-9
    return audio, onset_times


# ---------------------------------------------------------------------------
# Pitch Estimators & Handcrafted Confidence
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
    min_bin = np.argmin(np.abs(freqs - 80))
    max_bin = np.argmin(np.abs(freqs - 1000))
    bin_idx = np.argmax(mag[min_bin:max_bin]) + min_bin
    return float(freqs[bin_idx]), int(bin_idx)


def get_handcrafted_confidence(acf: np.ndarray, cep: np.ndarray, mag: np.ndarray,
                               sr: int, win: int) -> tuple[float, float, float]:
    min_lag = int(sr / 1000); max_lag = int(sr / 80)
    acf_peak = np.max(acf[min_lag:max_lag])
    acf_conf = float(np.clip((acf_peak / (acf[0] + 1e-10) - 0.15) / (0.75 - 0.15), 0.0, 1.0))

    cep_conf = float(np.clip(1.0 - (cep[0] - (-23.0)) / 25.4, 0.0, 1.0))

    freqs = np.fft.rfftfreq(2 * (len(mag) - 1), d=1/sr) if len(mag) > 1 else np.array([1])
    min_bin = np.argmin(np.abs(freqs - 80)); max_bin = np.argmin(np.abs(freqs - 1000))
    spec_range = mag[min_bin:max_bin]
    stft_peak = np.max(spec_range)
    stft_conf = float(np.clip((stft_peak - np.mean(spec_range)) / (stft_peak + 1e-10), 0.0, 1.0))

    return acf_conf, cep_conf, stft_conf


# ---------------------------------------------------------------------------
# Onset Score Functions
# ---------------------------------------------------------------------------

def stft_onset_score(mag_cur: np.ndarray, mag_prev: np.ndarray) -> float:
    return float(np.sum(np.maximum(mag_cur - mag_prev, 0.0)))


def acf_onset_score(acf_cur: np.ndarray, acf_prev: np.ndarray, min_lag: int, max_lag: int) -> float:
    def prominence(acf):
        if acf[0] < 1e-9: return 0.0
        return float(np.max(acf[min_lag:max_lag]) / acf[0])
    return float(max(prominence(acf_prev) - prominence(acf_cur), 0.0))


def cep_onset_score(cep_cur: np.ndarray, cep_prev: np.ndarray, min_q: int, max_q: int) -> float:
    peak_cur  = float(np.max(np.abs(cep_cur[min_q:max_q])))
    peak_prev = float(np.max(np.abs(cep_prev[min_q:max_q])))
    return float(abs(peak_cur - peak_prev))


# ---------------------------------------------------------------------------
# Feature Extraction (12 elements)
# ---------------------------------------------------------------------------

def extract_descriptors(frame: np.ndarray, acf: np.ndarray, cep: np.ndarray,
                        mag: np.ndarray, sr: int, n_fft: int = 2048) -> list[float]:
    min_lag = int(sr / 1000); max_lag = int(sr / 80)
    freqs = np.fft.rfftfreq(n_fft, d=1/sr)
    min_bin = np.argmin(np.abs(freqs - 80)); max_bin = np.argmin(np.abs(freqs - 1000))
    min_q = min_lag; max_q = max_lag

    # Spectral Entropy
    spec_range = mag[min_bin:max_bin]
    spec_sum = np.sum(spec_range)
    if spec_sum > 1e-9:
        p = spec_range / spec_sum
        p = np.clip(p, 1e-12, 1.0)
        spec_entropy = -np.sum(p * np.log2(p)) / np.log2(len(p))
    else:
        spec_entropy = 1.0

    # Spectral Flatness
    if spec_sum > 1e-9:
        log_mean = np.mean(np.log(spec_range + 1e-12))
        spec_flatness = np.exp(log_mean) / (np.mean(spec_range) + 1e-12)
    else:
        spec_flatness = 1.0

    stft_p, stft_peak_idx = estimate_pitch_stft(mag, sr, n_fft)

    # STFT Peak Strength & Prominence
    stft_peak_strength = mag[stft_peak_idx] / (spec_sum + 1e-12)
    stft_peak_prominence = max(0.0, float((mag[stft_peak_idx] - np.mean(spec_range)) / (mag[stft_peak_idx] + 1e-12)))

    acf_p, acf_peak_idx = estimate_pitch_acf(acf, sr)

    # ACF Peak Strength & Prominence
    acf_peak_strength = acf[acf_peak_idx] / (acf[0] + 1e-12)
    acf_range = acf[min_lag:max_lag]
    acf_peak_prominence = float(np.clip((acf[acf_peak_idx] - np.mean(acf_range)) / (np.max(acf_range) - np.min(acf_range) + 1e-10), 0.0, 1.0))

    cep_c0 = float(cep[0])
    cep_p, cep_peak_idx = estimate_pitch_cepstrum(cep, sr)

    # Cepstral Peak Strength & Prominence
    cep_peak_strength = float(np.abs(cep[cep_peak_idx]) / (np.abs(cep_c0) + 1e-10))
    cep_range = np.abs(cep[min_q:max_q])
    cep_peak_prominence = float(np.clip((np.abs(cep[cep_peak_idx]) - np.mean(cep_range)) / (np.max(cep_range) - np.min(cep_range) + 1e-10), 0.0, 1.0))

    # Zero Crossing Rate & Frame Log Energy
    zcr = float(np.mean(np.abs(np.diff(np.sign(frame)))) / 2.0)
    frame_log_energy = float(np.log(np.mean(frame**2) + 1e-10))

    return [
        spec_entropy, spec_flatness, stft_peak_strength, stft_peak_prominence,
        acf_peak_strength, acf_peak_prominence, cep_c0, cep_peak_strength,
        cep_peak_prominence, zcr, frame_log_energy, float(acf_peak_idx)
    ]


# ---------------------------------------------------------------------------
# Pitch Dataset & Training (Exp 025 logic)
# ---------------------------------------------------------------------------

def collect_pitch_data(audio: np.ndarray, true_pitch: np.ndarray, sr: int, win: int, hop: int) -> tuple[np.ndarray, np.ndarray]:
    features = []
    errors = []
    num_frames = (len(audio) - win) // hop + 1
    acf_lags_history = []

    for n in range(num_frames):
        start = n * hop
        frame_raw = audio[start:start + win]
        frame_windowed = frame_raw * np.hanning(win)

        acf = compute_acf(frame_windowed)
        cep = compute_cepstrum(frame_windowed)
        mag = compute_stft(frame_windowed, sr)
        if mag.ndim > 1: mag = np.mean(mag, axis=1)

        p_stft, _ = estimate_pitch_stft(mag, sr, 2 * (len(mag)-1))
        p_acf, acf_peak_idx = estimate_pitch_acf(acf, sr)
        p_cep, _ = estimate_pitch_cepstrum(cep, sr)

        acf_lags_history.append(float(acf_peak_idx))
        if len(acf_lags_history) > 3: acf_lags_history.pop(0)
        acf_jitter = float(np.var(acf_lags_history))

        feats = extract_descriptors(frame_raw, acf, cep, mag, sr, 2 * (len(mag)-1))
        feats[-1] = acf_jitter
        feats.append(1.0) # bias

        p_true = true_pitch[min(start + win // 2, len(true_pitch)-1)]
        def err(p_est):
            if p_est <= 0: return 12.0
            return min(12.0 * np.abs(np.log2(p_est / p_true)), 12.0)

        features.append(feats)
        errors.append([err(p_stft), err(p_acf), err(p_cep)])

    return np.array(features), np.array(errors)


def train_pitch_models(sr: int, win: int, hop: int) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    X_list, Y_list = [], []

    # Compile the 6 sweeps
    # Clean
    audio, true_p = synthesise_sweep(sr, 2.5, 180.0, 550.0, seed=10)
    X, Y = collect_pitch_data(audio, true_p, sr, win, hop)
    X_list.append(X); Y_list.append(Y)
    # Noisy
    audio, true_p = synthesise_sweep(sr, 2.5, 180.0, 550.0, seed=11)
    X, Y = collect_pitch_data(audio + np.random.normal(0, 0.15, len(audio)), true_p, sr, win, hop)
    X_list.append(X); Y_list.append(Y)
    # Filtered
    audio, true_p = synthesise_sweep(sr, 2.5, 180.0, 550.0, seed=12)
    X, Y = collect_pitch_data(lowpass_filter(audio, sr, 400.0), true_p, sr, win, hop)
    X_list.append(X); Y_list.append(Y)
    # Distorted
    audio, true_p = synthesise_sweep(sr, 2.5, 180.0, 550.0, vibrato=True, seed=13)
    X, Y = collect_pitch_data(hard_clip(audio, 0.08), true_p, sr, win, hop)
    X_list.append(X); Y_list.append(Y)
    # Vibrato Clean
    audio, true_p = synthesise_sweep(sr, 2.5, 180.0, 550.0, vibrato=True, seed=14)
    X, Y = collect_pitch_data(audio, true_p, sr, win, hop)
    X_list.append(X); Y_list.append(Y)
    # Vibrato Noisy
    audio, true_p = synthesise_sweep(sr, 2.5, 180.0, 550.0, vibrato=True, seed=15)
    X, Y = collect_pitch_data(audio + np.random.normal(0, 0.10, len(audio)), true_p, sr, win, hop)
    X_list.append(X); Y_list.append(Y)

    X_train = np.concatenate(X_list, axis=0)
    Y_train = np.concatenate(Y_list, axis=0)

    # Standardise
    X_feats = X_train[:, :-1]
    mu = np.mean(X_feats, axis=0)
    sigma = np.std(X_feats, axis=0) + 1e-8
    X_std = X_train.copy()
    X_std[:, :-1] = (X_feats - mu) / sigma

    weights = []
    I = np.eye(X_std.shape[1])
    I[-1, -1] = 0.0 # no regularisation on bias

    for idx in range(3):
        y = Y_train[:, idx]
        w = np.linalg.pinv(X_std.T @ X_std + 0.1 * I) @ X_std.T @ y
        weights.append(w)

    return weights, mu, sigma


# ---------------------------------------------------------------------------
# Onset Feature and Score Extraction
# ---------------------------------------------------------------------------

def make_onset_labels(num_frames: int, onset_times: list[float],
                      sr: int, hop: int, sigma_frames: float = 2.0) -> np.ndarray:
    labels = np.zeros(num_frames)
    for t in onset_times:
        centre = t * sr / hop
        for i in range(num_frames):
            labels[i] += np.exp(-0.5 * ((i - centre) / sigma_frames) ** 2)
    return np.clip(labels, 0.0, 1.0)


def collect_onset_features_and_scores(audio: np.ndarray, onset_times: list[float],
                                      sr: int, win: int, hop: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    min_lag = int(sr / 1000); max_lag = int(sr / 80)
    num_frames = (len(audio) - win) // hop + 1
    
    features = []
    onset_scores = []
    reactive_confs = []

    prev_stft = np.zeros(win + 1) # dummy dry run to establish size
    prev_acf = np.zeros(win)
    prev_cep = np.zeros(win)

    acf_lags_history = []

    for n in range(num_frames):
        start = n * hop
        frame_raw = audio[start:start + win]
        frame_windowed = frame_raw * np.hanning(win)

        acf = compute_acf(frame_windowed)
        cep = compute_cepstrum(frame_windowed)
        mag = compute_stft(frame_windowed, sr)
        if mag.ndim > 1: mag = np.mean(mag, axis=1)

        # Set sizes on first iteration
        if n == 0:
            prev_stft = np.zeros_like(mag)
            prev_acf = np.zeros_like(acf)
            prev_cep = np.zeros_like(cep)

        # Onset scores
        s_score = stft_onset_score(mag, prev_stft)
        a_score = acf_onset_score(acf, prev_acf, min_lag, max_lag)
        c_score = cep_onset_score(cep, prev_cep, min_lag, max_lag)

        # Dynamic descriptors
        _, acf_peak_idx = estimate_pitch_acf(acf, sr)
        acf_lags_history.append(float(acf_peak_idx))
        if len(acf_lags_history) > 3: acf_lags_history.pop(0)
        acf_jitter = float(np.var(acf_lags_history))

        # Extracted features
        feats = extract_descriptors(frame_raw, acf, cep, mag, sr, 2 * (len(mag)-1))
        feats[-1] = acf_jitter
        feats.append(1.0) # bias

        # Reactive confidences
        r_acf, r_cep, r_stft = get_handcrafted_confidence(acf, cep, mag, sr, win)

        features.append(feats)
        onset_scores.append([s_score, a_score, c_score])
        reactive_confs.append([r_stft, r_acf, r_cep])

        prev_stft = mag.copy()
        prev_acf = acf.copy()
        prev_cep = cep.copy()

    labels = make_onset_labels(num_frames, onset_times, sr, hop)
    return np.array(features), np.array(onset_scores), np.array(reactive_confs), labels


# ---------------------------------------------------------------------------
# Onset Training Baseline (Exp 024 logic)
# ---------------------------------------------------------------------------

def train_onset_meta_layer(sr: int, win: int, hop: int, mu_p: np.ndarray, sigma_p: np.ndarray) -> np.ndarray:
    """
    Trains the onset meta-layer on onset labels using the standardised features.
    """
    X_train_list, Y_train_list = [], []

    for seed in range(6):
        audio, onsets = synthesise_onset_sequence(sr, seed=seed)
        X, onset_scores, _, labels = collect_onset_features_and_scores(audio, onsets, sr, win, hop)

        # Standardise using pitch statistics
        X_std = X.copy()
        X_std[:, :-1] = (X[:, :-1] - mu_p) / sigma_p

        # Normalise onset scores
        s_max = np.max(onset_scores[:, 0]) or 1.0
        a_max = np.max(onset_scores[:, 1]) or 1.0
        c_max = np.max(onset_scores[:, 2]) or 1.0
        s_sc = onset_scores[:, 0] / s_max
        a_sc = onset_scores[:, 1] / a_max
        c_sc = onset_scores[:, 2] / c_max

        for i in range(len(X)):
            lbl = labels[i]
            w_s = (s_sc[i] * lbl) + 1e-6
            w_a = (a_sc[i] * lbl) + 1e-6
            w_c = (c_sc[i] * lbl) + 1e-6
            total = w_s + w_a + w_c
            X_train_list.append(X_std[i])
            Y_train_list.append([w_s / total, w_a / total, w_c / total])

    # Moore-Penrose pseudo-inverse solve
    W_onset = np.linalg.pinv(np.array(X_train_list)) @ np.array(Y_train_list)
    return W_onset


# ---------------------------------------------------------------------------
# Peak Picking & Evaluation
# ---------------------------------------------------------------------------

def pick_peaks(scores: np.ndarray, hop: int, sr: int,
               threshold: float = 0.35, min_ioi_ms: float = 80.0) -> list[float]:
    min_gap = int(min_ioi_ms * sr / (hop * 1000))
    peaks, _ = scipy.signal.find_peaks(scores, height=threshold, distance=max(min_gap, 1))
    return [int(p) * hop / sr for p in peaks]


def evaluate(detected: list[float], reference: list[float], tol_s: float = 0.050) -> dict:
    matched_ref = set()
    tp = 0
    for d in detected:
        for i, r in enumerate(reference):
            if i not in matched_ref and abs(d - r) <= tol_s:
                tp += 1
                matched_ref.add(i)
                break
    fp = len(detected) - tp
    fn = len(reference) - len(matched_ref)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {"P": prec, "R": rec, "F1": f1}


# ---------------------------------------------------------------------------
# Running Fusions
# ---------------------------------------------------------------------------

def run_onset_evaluation(audio: np.ndarray, onset_times: list[float], sr: int, win: int, hop: int,
                          W_pitch: list[np.ndarray], W_onset: np.ndarray,
                          mu_p: np.ndarray, sigma_p: np.ndarray) -> dict:
    
    X, onset_scores, reactive_confs, _ = collect_onset_features_and_scores(audio, onset_times, sr, win, hop)
    t = np.array([(n * hop + win // 2) / sr for n in range(len(X))])

    # Normalise onset scores individually
    s_scores = onset_scores[:, 0] / (np.max(onset_scores[:, 0]) or 1.0)
    a_scores = onset_scores[:, 1] / (np.max(onset_scores[:, 1]) or 1.0)
    c_scores = onset_scores[:, 2] / (np.max(onset_scores[:, 2]) or 1.0)

    # Standardise features with pitch stats
    X_std = X.copy()
    X_std[:, :-1] = (X[:, :-1] - mu_p) / sigma_p

    # 1. Reactive fusion
    rw_sum = reactive_confs.sum(axis=1, keepdims=True) + 1e-10
    rw = reactive_confs / rw_sum
    fused_reactive = rw[:, 0] * s_scores + rw[:, 1] * a_scores + rw[:, 2] * c_scores
    fused_reactive /= (np.max(fused_reactive) or 1.0)

    # 2. Onset-Trained Meta-layer
    y_onset_pred = X_std @ W_onset
    w_onset_pred = np.maximum(y_onset_pred, 0.0)
    w_onset_pred /= (w_onset_pred.sum(axis=1, keepdims=True) + 1e-10)
    fused_onset_meta = w_onset_pred[:, 0] * s_scores + w_onset_pred[:, 1] * a_scores + w_onset_pred[:, 2] * c_scores
    fused_onset_meta /= (np.max(fused_onset_meta) or 1.0)

    # 3. Cross-Task Transferred (pitch models predicting pitch errors)
    err_stft = np.maximum(X_std @ W_pitch[0], 0.0)
    err_acf  = np.maximum(X_std @ W_pitch[1], 0.0)
    err_cep  = np.maximum(X_std @ W_pitch[2], 0.0)

    eps = 0.05
    w_ct_stft = 1.0 / (err_stft + eps)
    w_ct_acf  = 1.0 / (err_acf  + eps)
    w_ct_cep  = 1.0 / (err_cep  + eps)
    w_ct_sum  = w_ct_stft + w_ct_acf + w_ct_cep + 1e-10
    
    w_ct_stft /= w_ct_sum
    w_ct_acf  /= w_ct_sum
    w_ct_cep  /= w_ct_sum

    fused_ct = w_ct_stft * s_scores + w_ct_acf * a_scores + w_ct_cep * c_scores
    fused_ct /= (np.max(fused_ct) or 1.0)

    # Compile curves
    curves = {
        "STFT":     s_scores,
        "ACF":      a_scores,
        "Cepstrum": c_scores,
        "Reactive": fused_reactive,
        "Meta":     fused_onset_meta,
        "Cross":    fused_ct
    }

    # Evaluate F1
    results = {}
    for name, scores in curves.items():
        detected = pick_peaks(scores, hop, sr, threshold=0.35)
        eval_dict = evaluate(detected, onset_times)
        results[name] = {**eval_dict, "scores": scores}

    results["_t"] = t
    results["_ct_weights"] = np.column_stack([w_ct_stft, w_ct_acf, w_ct_cep])
    results["_meta_weights"] = w_onset_pred
    results["_pred_pitch_errs"] = np.column_stack([err_stft, err_acf, err_cep])
    return results


# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------

def run():
    print("=" * 60)
    print("EXPERIMENT 026 — CROSS-TASK TRANSFER OF FAILURE MANIFOLD")
    print("=" * 60)

    sr = 22050
    WIN = 1024
    HOP = 256

    # 1. Train pitch models
    print("Training failure models on pitch tracking data (Exp 025 sweeps)...")
    W_pitch, mu_p, sigma_p = train_pitch_models(sr, WIN, HOP)
    print("Pitch failure models trained.\n")

    # 2. Train onset baseline meta-layer
    print("Training task-specific onset meta-layer baseline (Exp 024 sequences)...")
    W_onset = train_onset_meta_layer(sr, WIN, HOP, mu_p, sigma_p)
    print("Onset baseline meta-layer trained.\n")

    # 3. Generate onset test signal (note sequence)
    print("Synthesising onset test sequence...")
    audio_clean, onset_times = synthesise_onset_sequence(sr, seed=99)
    print(f"Test sequence: {len(onset_times)} notes, {len(audio_clean)/sr:.2f}s total")

    test_conditions = {
        "Clean": audio_clean,
        "Noisy (σ=0.15)": audio_clean + np.random.normal(0, 0.15, len(audio_clean)),
        "Filtered (LP=400Hz)": lowpass_filter(audio_clean, sr, cutoff=400.0),
        "Distorted (clip=0.08)": hard_clip(audio_clean, threshold=0.08)
    }

    # Evaluate all conditions
    all_results = {}
    methods = ["STFT", "ACF", "Cepstrum", "Reactive", "Meta", "Cross"]

    print(f"{'Condition':<22}  {'Detector':<10}  {'P':>6}  {'R':>6}  {'F1':>6}")
    print("-" * 56)

    for cond_name, audio in test_conditions.items():
        res = run_onset_evaluation(audio, onset_times, sr, WIN, HOP, W_pitch, W_onset, mu_p, sigma_p)
        all_results[cond_name] = res
        for name in methods:
            r = res[name]
            marker = " ★" if name == "Cross" else " ◀" if name == "Meta" else ""
            print(f"{cond_name:<22}  {name:<10}  {r['P']:>6.3f}  {r['R']:>6.3f}  {r['F1']:>6.3f}{marker}")
        print()

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    print("Generating transferability plots...")
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(16, 15))
    gs = fig.add_gridspec(4, 2, hspace=0.35, wspace=0.25)

    # We will show weights and onset curve diagnostics for the "Noisy" and "Filtered" conditions
    # to inspect dynamic routing details.
    
    # 1. Noisy Condition - Fused Onset Scores comparison
    ax_noisy_curves = fig.add_subplot(gs[0, 0])
    noisy_res = all_results["Noisy (σ=0.15)"]
    t = noisy_res["_t"]
    ax_noisy_curves.plot(t, noisy_res["STFT"]["scores"], color="#a6d854", alpha=0.3, label="STFT Flux")
    ax_noisy_curves.plot(t, noisy_res["Cepstrum"]["scores"], color="#e7298a", alpha=0.3, label="Cep Velocity")
    ax_noisy_curves.plot(t, noisy_res["Cross"]["scores"], color="#33cc66", lw=1.8, label="Cross-Task Transferred")
    ax_noisy_curves.plot(t, noisy_res["Meta"]["scores"], color="cyan", lw=1.0, linestyle="--", label="Onset-Trained Meta")
    for ot in onset_times:
        ax_noisy_curves.axvline(ot, color="white", alpha=0.2, linestyle=":")
    ax_noisy_curves.set_title("Fused Onset Scores comparison (Noisy Condition)", fontsize=10, fontweight="bold")
    ax_noisy_curves.set_ylabel("Onset Strength (Normalised)")
    ax_noisy_curves.legend(fontsize=8)
    ax_noisy_curves.grid(True, alpha=0.1)

    # 2. Noisy Condition - Dynamic Weights (Transferred pitch model)
    ax_noisy_weights = fig.add_subplot(gs[0, 1])
    ct_w = noisy_res["_ct_weights"]
    ax_noisy_weights.stackplot(t, ct_w[:, 0], ct_w[:, 1], ct_w[:, 2],
                              labels=["STFT Weight", "ACF Weight", "Cep Weight"],
                              colors=["#a6d854", "#377eb8", "#e7298a"], alpha=0.7)
    ax_noisy_weights.set_title("Cross-Task Transferred Weights (Noisy Condition)", fontsize=10, fontweight="bold")
    ax_noisy_weights.set_ylim(0, 1.0)
    ax_noisy_weights.legend(fontsize=8, loc="lower left")
    ax_noisy_weights.grid(True, alpha=0.1)

    # 3. Filtered Condition - Fused Onset Scores comparison
    ax_filt_curves = fig.add_subplot(gs[1, 0])
    filt_res = all_results["Filtered (LP=400Hz)"]
    ax_filt_curves.plot(t, filt_res["STFT"]["scores"], color="#a6d854", alpha=0.3, label="STFT Flux")
    ax_filt_curves.plot(t, filt_res["Cepstrum"]["scores"], color="#e7298a", alpha=0.3, label="Cep Velocity")
    ax_filt_curves.plot(t, filt_res["Cross"]["scores"], color="#33cc66", lw=1.8, label="Cross-Task Transferred")
    ax_filt_curves.plot(t, filt_res["Meta"]["scores"], color="cyan", lw=1.0, linestyle="--", label="Onset-Trained Meta")
    for ot in onset_times:
        ax_filt_curves.axvline(ot, color="white", alpha=0.2, linestyle=":")
    ax_filt_curves.set_title("Fused Onset Scores comparison (Filtered Condition)", fontsize=10, fontweight="bold")
    ax_filt_curves.set_ylabel("Onset Strength (Normalised)")
    ax_filt_curves.legend(fontsize=8)
    ax_filt_curves.grid(True, alpha=0.1)

    # 4. Filtered Condition - Dynamic Weights (Transferred pitch model)
    ax_filt_weights = fig.add_subplot(gs[1, 1])
    ct_w_f = filt_res["_ct_weights"]
    ax_filt_weights.stackplot(t, ct_w_f[:, 0], ct_w_f[:, 1], ct_w_f[:, 2],
                             labels=["STFT Weight", "ACF Weight", "Cep Weight"],
                             colors=["#a6d854", "#377eb8", "#e7298a"], alpha=0.7)
    ax_filt_weights.set_title("Cross-Task Transferred Weights (Filtered Condition)", fontsize=10, fontweight="bold")
    ax_filt_weights.set_ylim(0, 1.0)
    ax_filt_weights.legend(fontsize=8, loc="lower left")
    ax_filt_weights.grid(True, alpha=0.1)

    # 5. Distorted Condition - Fused Onset Scores comparison
    ax_dist_curves = fig.add_subplot(gs[2, 0])
    dist_res = all_results["Distorted (clip=0.08)"]
    ax_dist_curves.plot(t, dist_res["STFT"]["scores"], color="#a6d854", alpha=0.3, label="STFT Flux")
    ax_dist_curves.plot(t, dist_res["Cepstrum"]["scores"], color="#e7298a", alpha=0.3, label="Cep Velocity")
    ax_dist_curves.plot(t, dist_res["Cross"]["scores"], color="#33cc66", lw=1.8, label="Cross-Task Transferred")
    ax_dist_curves.plot(t, dist_res["Meta"]["scores"], color="cyan", lw=1.0, linestyle="--", label="Onset-Trained Meta")
    for ot in onset_times:
        ax_dist_curves.axvline(ot, color="white", alpha=0.2, linestyle=":")
    ax_dist_curves.set_title("Fused Onset Scores comparison (Distorted Condition)", fontsize=10, fontweight="bold")
    ax_dist_curves.set_ylabel("Onset Strength (Normalised)")
    ax_dist_curves.legend(fontsize=8)
    ax_dist_curves.grid(True, alpha=0.1)

    # 6. Distorted Condition - Dynamic Weights (Transferred pitch model)
    ax_dist_weights = fig.add_subplot(gs[2, 1])
    ct_w_d = dist_res["_ct_weights"]
    ax_dist_weights.stackplot(t, ct_w_d[:, 0], ct_w_d[:, 1], ct_w_d[:, 2],
                             labels=["STFT Weight", "ACF Weight", "Cep Weight"],
                             colors=["#a6d854", "#377eb8", "#e7298a"], alpha=0.7)
    ax_dist_weights.set_title("Cross-Task Transferred Weights (Distorted Condition)", fontsize=10, fontweight="bold")
    ax_dist_weights.set_ylim(0, 1.0)
    ax_dist_weights.legend(fontsize=8, loc="lower left")
    ax_dist_weights.grid(True, alpha=0.1)

    # 7. Summary bar chart comparing onset F1-scores across all conditions
    ax_bar = fig.add_subplot(gs[3, :])
    det_names = ["STFT", "ACF", "Cepstrum", "Reactive", "Onset-Trained Meta", "Cross-Task Transferred"]
    det_colors = ["#a6d854", "#377eb8", "#e7298a", "#ff7f00", "cyan", "#33cc66"]
    
    cond_names = list(test_conditions.keys())
    x = np.arange(len(cond_names))
    width = 0.12
    offsets = np.linspace(-2.5, 2.5, 6) * width

    for d_idx, name in enumerate(det_names):
        f1s = []
        for cond in cond_names:
            # map name to result keys
            key = "Meta" if name == "Onset-Trained Meta" else "Cross" if name == "Cross-Task Transferred" else name
            f1s.append(all_results[cond][key]["F1"])
            
        bars = ax_bar.bar(x + offsets[d_idx], f1s, width * 0.9,
                          label=name, color=det_colors[d_idx], alpha=0.85)
        for bar in bars:
            val = bar.get_height()
            ax_bar.text(bar.get_x() + bar.get_width()/2, val + 0.01,
                        f"{val:.2f}", ha="center", va="bottom", fontsize=7, color="white")

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(cond_names, fontsize=9)
    ax_bar.set_ylabel("F1 Score")
    ax_bar.set_ylim(0, 1.2)
    ax_bar.set_title("Onset Detection F1 Score Comparison: Universality of Failure Geometry", fontsize=11, fontweight="bold")
    ax_bar.legend(fontsize=9, loc="upper right")
    ax_bar.grid(True, alpha=0.1, axis="y")

    fig.suptitle("Experiment 026 — Cross-Task Transfer of Failure Manifold (Pitch → Onsets)", fontsize=14, fontweight="bold", y=0.98)
    
    out_path = os.path.join(project_root, "results", "exp026_cross_task_transfer.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    run()
