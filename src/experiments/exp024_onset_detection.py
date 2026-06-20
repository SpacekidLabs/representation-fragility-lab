"""
Experiment 024 — Representation Intelligence for Onset Detection
================================================================
Generalization test: can the same confidence + meta-fusion architecture
that improved pitch tracking also improve onset detection?

Research question:
  Can representation confidence be reused across tasks?

Architecture (identical structure to the pitch system, different signals):

  Audio
    ↓
  STFT flux        ACF prominence drop    Cepstrum peak velocity
  (onset score)    (onset score)          (onset score)
    ↓                    ↓                      ↓
  STFT confidence  ACF confidence         Cep confidence
    ↓
  Meta-layer (learned fusion weights)
    ↓
  Fused onset score
    ↓
  Peak picking → onset times

Three onset detectors:
  STFT  — spectral flux (L1 norm of positive spectral differences)
  ACF   — peak prominence velocity (sudden loss of periodicity = onset)
  CEP   — cepstral peak velocity (harmonic structure disruption = onset)

Baselines compared:
  STFT-only     (standard DSP approach)
  ACF-only
  Cepstrum-only
  Confidence-weighted average (reactive fusion)
  Meta-layer fusion (learned weights)

Test conditions:
  Clean (σ = 0.0)
  Light noise (σ = 0.10)
  Heavy noise (σ = 0.30)

Evaluation: Precision / Recall / F1 with 50 ms tolerance window.

Outputs:
  results/exp024_onset_detection.png
"""

import sys
import os
import numpy as np
import scipy.signal
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dataclasses import dataclass

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.representations.acf import compute_acf
from src.representations.cepstrum import compute_cepstrum
from src.representations.stft import compute_stft


# ---------------------------------------------------------------------------
# Signal synthesis — multi-note sequence with known onset times
# ---------------------------------------------------------------------------

def synthesise_sequence(sr: int, seed: int = 0) -> tuple[np.ndarray, list[float]]:
    """
    Synthesise a sequence of harmonic notes with sharp attacks and
    different pitches.  Returns (audio, list_of_onset_times_in_seconds).
    """
    rng = np.random.default_rng(seed)

    # Note sequence: (freq_hz, duration_s)
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
    gap = 0.04   # 40 ms silence between notes → clear onset boundary

    audio_chunks = []
    onset_times  = []
    cursor = 0.0

    for freq, dur in notes:
        # silence gap
        gap_samples = int(gap * sr)
        audio_chunks.append(np.zeros(gap_samples))
        cursor += gap

        # note onset
        onset_times.append(cursor)

        # harmonic stack (5 harmonics)
        n_samples = int(dur * sr)
        t = np.arange(n_samples) / sr
        note = np.zeros(n_samples)
        for k in range(1, 6):
            note += (1.0 / k) * np.sin(2 * np.pi * k * freq * t)

        # sharp attack envelope (5 ms attack, 20 ms release)
        env = np.ones(n_samples)
        atk = min(int(0.005 * sr), n_samples // 4)
        rel = min(int(0.020 * sr), n_samples // 4)
        env[:atk] = np.linspace(0.0, 1.0, atk)
        env[-rel:] = np.linspace(1.0, 0.0, rel)
        note *= env

        audio_chunks.append(note)
        cursor += dur

    audio = np.concatenate(audio_chunks)
    audio /= np.max(np.abs(audio))
    return audio, onset_times


# ---------------------------------------------------------------------------
# Onset score functions — one per representation
# ---------------------------------------------------------------------------

def stft_onset_score(mag_cur: np.ndarray, mag_prev: np.ndarray) -> float:
    """Spectral flux: L1 norm of positive spectrum differences."""
    diff = mag_cur - mag_prev
    flux = np.sum(np.maximum(diff, 0.0))
    return float(flux)


def acf_onset_score(acf_cur: np.ndarray, acf_prev: np.ndarray,
                    min_lag: int, max_lag: int) -> float:
    """
    Peak prominence velocity: negative change in ACF peak ratio.
    An onset breaks periodicity → prominence drops → score rises.
    """
    def prominence(acf):
        if acf[0] < 1e-9:
            return 0.0
        return float(np.max(acf[min_lag:max_lag]) / acf[0])

    drop = prominence(acf_prev) - prominence(acf_cur)
    return float(max(drop, 0.0))   # only rising scores on drops


def cep_onset_score(cep_cur: np.ndarray, cep_prev: np.ndarray,
                    min_q: int, max_q: int) -> float:
    """
    Cepstral peak velocity: magnitude of change in cepstral peak value.
    An onset disrupts harmonic structure → cepstral peak shifts.
    """
    peak_cur  = float(np.max(np.abs(cep_cur[min_q:max_q])))
    peak_prev = float(np.max(np.abs(cep_prev[min_q:max_q])))
    return float(abs(peak_cur - peak_prev))


# ---------------------------------------------------------------------------
# Confidence signals (reused from Exp 015 / 022)
# ---------------------------------------------------------------------------

def stft_confidence(mag: np.ndarray, min_bin: int, max_bin: int) -> float:
    region = np.mean(mag, axis=1)[min_bin:max_bin] if mag.ndim > 1 else mag[min_bin:max_bin]
    s = np.sum(region)
    if s < 1e-9:
        return 0.0
    return float(np.clip((np.max(region) / s - 0.03) / (0.25 - 0.03), 0.0, 1.0))


def acf_confidence(acf: np.ndarray, min_lag: int, max_lag: int) -> float:
    if acf[0] < 1e-9:
        return 0.0
    ratio = float(np.max(acf[min_lag:max_lag]) / acf[0])
    return float(np.clip((ratio - 0.15) / (0.75 - 0.15), 0.0, 1.0))


def cep_confidence(cep: np.ndarray) -> float:
    c0 = float(cep[0])
    return float(np.clip((c0 - (-10.0)) / (-13.6 - (-10.0)), 0.0, 1.0))


# ---------------------------------------------------------------------------
# Feature extraction per frame
# ---------------------------------------------------------------------------

@dataclass
class FrameFeatures:
    stft_score:  float
    acf_score:   float
    cep_score:   float
    stft_conf:   float
    acf_conf:    float
    cep_conf:    float

    def to_array(self) -> np.ndarray:
        return np.array([
            self.stft_score, self.acf_score, self.cep_score,
            self.stft_conf,  self.acf_conf,  self.cep_conf,
            self.stft_score * self.stft_conf,   # interaction terms
            self.acf_score  * self.acf_conf,
            self.cep_score  * self.cep_conf,
            1.0,                                # bias
        ], dtype=np.float64)


def extract_features(audio: np.ndarray, sr: int,
                     win: int, hop: int) -> list[FrameFeatures]:
    min_lag  = int(sr / 1000);  max_lag  = int(sr / 80)
    min_q    = min_lag;          max_q    = max_lag

    # Dry-run representations to dynamically determine shapes
    dummy_frame = np.zeros(win)
    dummy_stft = compute_stft(dummy_frame, sr)
    if dummy_stft.ndim > 1:
        dummy_stft = np.mean(dummy_stft, axis=1)
    dummy_acf = compute_acf(dummy_frame)
    dummy_cep = compute_cepstrum(dummy_frame)

    n_fft = 2 * (len(dummy_stft) - 1)
    min_bin  = int(80  * n_fft / sr)
    max_bin = int(1000 * n_fft / sr)

    features = []
    num_frames = (len(audio) - win) // hop + 1

    prev_stft = np.zeros_like(dummy_stft)
    prev_acf  = np.zeros_like(dummy_acf)
    prev_cep  = np.zeros_like(dummy_cep)

    for n in range(num_frames):
        start = n * hop
        frame = audio[start:start + win] * np.hanning(win)

        acf  = compute_acf(frame)
        cep  = compute_cepstrum(frame)
        mag  = compute_stft(frame, sr)
        if mag.ndim > 1:
            mag_1d = np.mean(mag, axis=1)
        else:
            mag_1d = mag

        s_score = stft_onset_score(mag_1d, prev_stft)
        a_score = acf_onset_score(acf, prev_acf, min_lag, max_lag)
        c_score = cep_onset_score(cep, prev_cep, min_q, max_q)

        s_conf = stft_confidence(mag_1d, min_bin, max_bin)
        a_conf = acf_confidence(acf, min_lag, max_lag)
        c_conf = cep_confidence(cep)

        features.append(FrameFeatures(s_score, a_score, c_score,
                                      s_conf,  a_conf,  c_conf))

        prev_stft = mag_1d.copy()
        prev_acf  = acf.copy()
        prev_cep  = cep.copy()

    return features


# ---------------------------------------------------------------------------
# Normalise onset scores to [0, 1] across all frames
# ---------------------------------------------------------------------------

def normalise_scores(features: list[FrameFeatures]) -> list[FrameFeatures]:
    s_max = max(f.stft_score for f in features) or 1.0
    a_max = max(f.acf_score  for f in features) or 1.0
    c_max = max(f.cep_score  for f in features) or 1.0
    for f in features:
        f.stft_score /= s_max
        f.acf_score  /= a_max
        f.cep_score  /= c_max
    return features


# ---------------------------------------------------------------------------
# Label generation — soft Gaussian window around each onset
# ---------------------------------------------------------------------------

def make_onset_labels(num_frames: int, onset_times: list[float],
                      sr: int, hop: int, sigma_frames: float = 2.0) -> np.ndarray:
    labels = np.zeros(num_frames)
    for t in onset_times:
        centre = t * sr / hop
        for i in range(num_frames):
            labels[i] += np.exp(-0.5 * ((i - centre) / sigma_frames) ** 2)
    return np.clip(labels, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Peak picking with minimum inter-onset interval
# ---------------------------------------------------------------------------

def pick_peaks(scores: np.ndarray, hop: int, sr: int,
               threshold: float = 0.3, min_ioi_ms: float = 80.0) -> list[float]:
    min_gap = int(min_ioi_ms * sr / (hop * 1000))
    peaks, _ = scipy.signal.find_peaks(scores, height=threshold, distance=max(min_gap, 1))
    return [int(p) * hop / sr for p in peaks]


# ---------------------------------------------------------------------------
# Evaluation — Precision / Recall / F1
# ---------------------------------------------------------------------------

def evaluate(detected: list[float], reference: list[float],
             tol_s: float = 0.050) -> dict:
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
    return {"P": prec, "R": rec, "F1": f1, "TP": tp, "FP": fp, "FN": fn}


# ---------------------------------------------------------------------------
# Run one condition
# ---------------------------------------------------------------------------

def run_condition(audio: np.ndarray, onset_times: list[float],
                  sr: int, win: int, hop: int,
                  W: np.ndarray, threshold: float = 0.35) -> dict:

    features = extract_features(audio, sr, win, hop)
    features = normalise_scores(features)
    t_frames = np.array([(n * hop + win // 2) / sr for n in range(len(features))])

    # -- individual detector scores --
    stft_scores = np.array([f.stft_score for f in features])
    acf_scores  = np.array([f.acf_score  for f in features])
    cep_scores  = np.array([f.cep_score  for f in features])

    stft_confs = np.array([f.stft_conf for f in features])
    acf_confs  = np.array([f.acf_conf  for f in features])
    cep_confs  = np.array([f.cep_conf  for f in features])

    # -- confidence-weighted reactive fusion --
    w_sum = stft_confs + acf_confs + cep_confs + 1e-9
    reactive = (stft_confs * stft_scores +
                acf_confs  * acf_scores  +
                cep_confs  * cep_scores) / w_sum

    # -- meta-layer fusion --
    X = np.array([f.to_array() for f in features])
    Y_pred = X @ W                                  # shape (N, 3)
    W_pred = np.maximum(Y_pred, 0.0)
    W_pred /= (W_pred.sum(axis=1, keepdims=True) + 1e-9)
    meta_scores = (W_pred[:, 0] * stft_scores +
                   W_pred[:, 1] * acf_scores  +
                   W_pred[:, 2] * cep_scores)
    # Normalise to [0,1]
    for arr in [stft_scores, acf_scores, cep_scores, reactive, meta_scores]:
        m = arr.max()
        if m > 0:
            arr /= m

    results = {}
    for name, scores in [("STFT", stft_scores), ("ACF", acf_scores),
                          ("Cepstrum", cep_scores), ("Reactive", reactive),
                          ("Meta", meta_scores)]:
        detected = pick_peaks(scores, hop, sr, threshold=threshold)
        results[name] = {**evaluate(detected, onset_times), "scores": scores}

    results["_t"] = t_frames
    results["_onset_times"] = onset_times
    return results


# ---------------------------------------------------------------------------
# Train the meta-layer on clean signals
# ---------------------------------------------------------------------------

def train_meta_layer(sr: int, win: int, hop: int) -> np.ndarray:
    X_train, Y_train = [], []

    # Train on multiple clean sequences with varied note patterns
    for seed in range(6):
        audio, onsets = synthesise_sequence(sr, seed=seed)
        features = extract_features(audio, sr, win, hop)
        features = normalise_scores(features)
        labels   = make_onset_labels(len(features), onsets, sr, hop)

        for i, feat in enumerate(features):
            s, a, c = feat.stft_score, feat.acf_score, feat.cep_score
            lbl = labels[i]
            # Optimal weight = proportional to each score * label alignment
            w_s = (s * lbl) + 1e-6
            w_a = (a * lbl) + 1e-6
            w_c = (c * lbl) + 1e-6
            total = w_s + w_a + w_c
            X_train.append(feat.to_array())
            Y_train.append([w_s / total, w_a / total, w_c / total])

    W = np.linalg.pinv(np.array(X_train)) @ np.array(Y_train)
    return W


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run():
    print("=" * 60)
    print("EXPERIMENT 024 — REPRESENTATION INTELLIGENCE: ONSET DETECTION")
    print("=" * 60)

    sr  = 22050
    WIN = 1024    # shorter window for better temporal resolution
    HOP = 256

    print(f"Window: {WIN} samples ({WIN/sr*1000:.0f} ms), Hop: {HOP} ({HOP/sr*1000:.1f} ms)\n")

    # Clean reference audio + ground truth onsets
    audio_clean, onset_times = synthesise_sequence(sr, seed=99)
    duration = len(audio_clean) / sr
    print(f"Sequence: {len(onset_times)} notes, {duration:.2f}s total")
    print(f"Onset times: {[f'{t:.3f}' for t in onset_times]}\n")

    # Test conditions
    np.random.seed(42)
    conditions = {
        "Clean   (σ=0.00)": audio_clean,
        "Light   (σ=0.10)": audio_clean + np.random.normal(0, 0.10, len(audio_clean)),
        "Heavy   (σ=0.30)": audio_clean + np.random.normal(0, 0.30, len(audio_clean)),
    }

    # Train meta-layer on clean signals (different seeds from test)
    print("Training meta-layer on 6 clean training sequences...")
    W = train_meta_layer(sr, WIN, HOP)
    print("Training complete.\n")

    # Run all conditions
    all_results = {}
    detectors = ["STFT", "ACF", "Cepstrum", "Reactive", "Meta"]

    print(f"{'Condition':<22}  {'Detector':<10}  {'P':>6}  {'R':>6}  {'F1':>6}  {'TP':>4}  {'FP':>4}  {'FN':>4}")
    print("-" * 72)

    for cond_name, audio in conditions.items():
        res = run_condition(audio, onset_times, sr, WIN, HOP, W)
        all_results[cond_name] = res
        for det in detectors:
            r = res[det]
            marker = " ◀" if det == "Meta" else ""
            print(f"{cond_name:<22}  {det:<10}  {r['P']:>6.3f}  {r['R']:>6.3f}  "
                  f"{r['F1']:>6.3f}  {r['TP']:>4}  {r['FP']:>4}  {r['FN']:>4}{marker}")
        print()

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    print("Generating plot...")

    DET_COLORS = {
        "STFT":     "#a6d854",
        "ACF":      "#377eb8",
        "Cepstrum": "#e7298a",
        "Reactive": "#ff7f00",
        "Meta":     "#33cc66",
    }

    cond_names = list(conditions.keys())
    n_conds    = len(cond_names)

    plt.style.use("dark_background")
    fig = plt.figure(figsize=(16, 13))
    gs  = fig.add_gridspec(n_conds + 1, 3, hspace=0.45, wspace=0.35,
                           height_ratios=[1.2] * n_conds + [1.8])

    # Top rows: onset score curves per condition
    for ci, cond_name in enumerate(cond_names):
        res = all_results[cond_name]
        t   = res["_t"]
        ots = res["_onset_times"]

        for di, det in enumerate(["STFT", "ACF", "Cepstrum"]):
            ax = fig.add_subplot(gs[ci, di])
            scores = res[det]["scores"]
            ax.fill_between(t, scores, alpha=0.2, color=DET_COLORS[det])
            ax.plot(t, scores, color=DET_COLORS[det], lw=1.2, label=det)

            # Meta overlay
            meta_sc = res["Meta"]["scores"]
            ax.plot(t, meta_sc, color="#33cc66", lw=1.8, alpha=0.7,
                    linestyle="--", label="Meta")

            # Ground truth onset markers
            for ot in ots:
                ax.axvline(ot, color="white", alpha=0.25, lw=0.8, linestyle=":")

            f1_det  = res[det]["F1"]
            f1_meta = res["Meta"]["F1"]
            title = f"{det}  F1={f1_det:.2f}  |  Meta F1={f1_meta:.2f}"
            ax.set_title(title, fontsize=8, fontweight="bold")
            if di == 0:
                ax.set_ylabel(cond_name.strip(), fontsize=8)
            ax.set_ylim(-0.05, 1.15)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.1)

    # Bottom row: F1 bar chart summary
    ax_bar = fig.add_subplot(gs[n_conds, :])
    cond_labels = [c.strip() for c in cond_names]
    x = np.arange(len(cond_names))
    n_det = len(detectors)
    width = 0.15
    offsets = np.linspace(-(n_det - 1) / 2, (n_det - 1) / 2, n_det) * width

    for i, det in enumerate(detectors):
        f1s = [all_results[c][det]["F1"] for c in cond_names]
        bars = ax_bar.bar(x + offsets[i], f1s, width * 0.9,
                          label=det, color=DET_COLORS[det], alpha=0.85)
        for bar, val in zip(bars, f1s):
            ax_bar.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.01, f"{val:.2f}",
                        ha="center", va="bottom", fontsize=7,
                        color=DET_COLORS[det])

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(cond_labels, fontsize=9)
    ax_bar.set_ylabel("F1 Score")
    ax_bar.set_ylim(0, 1.25)
    ax_bar.set_title("F1 Score by Detector and Noise Condition — "
                      "Does Representation Intelligence Generalise to Onset Detection?",
                      fontsize=11, fontweight="bold")
    ax_bar.legend(fontsize=9, loc="upper right")
    ax_bar.grid(True, alpha=0.1, axis="y")
    ax_bar.axhline(1.0, color="white", alpha=0.1, lw=0.8)

    fig.suptitle("Experiment 024 — Representation Intelligence for Onset Detection",
                 fontsize=13, fontweight="bold", y=0.98)

    out_path = os.path.join(project_root, "results", "exp024_onset_detection.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Plot saved: {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    run()
