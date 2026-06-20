"""
Experiment 036 — State-Space Theory Formalisation
=================================================
Formally evaluates the Local State Hypothesis by computing the State 
Compatibility Index (eta-squared) for ten different DSP tasks.

Outputs:
  results/exp036_theory_validation.png
"""

import sys
import os
import warnings

import numpy as np
import scipy.signal
import scipy.fftpack
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold

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


# ──────────────────────────────────────────────────────────────────────────────
# DATASET SYNTHESIS
# ──────────────────────────────────────────────────────────────────────────────

def synthesize_validation_dataset(engine, n_frames=500):
    """
    Synthesizes a diverse set of 500 audio frames (1024 samples, 16kHz)
    spanning variations in signal class, level, noise, decay, EQ, timbre, and beats.
    
    Returns:
        coords: np.ndarray, shape (n_frames, 2) of (z1, z2) coordinates
        targets: dict mapping task name -> 1D np.ndarray of target values
    """
    sr = 16000
    N = 1024
    t = np.arange(N) / sr
    
    rng = np.random.default_rng(12345)
    
    coords = []
    
    # Task target arrays
    y_pitch = []     # Binary: 1 = needs 4096 window (low f0 or high noise), 0 = 1024
    y_voicing = []   # Binary: 1 = voiced periodic, 0 = unvoiced/noise
    y_onset = []     # Binary: 1 = onset hit in frame, 0 = no onset
    y_denoising = [] # Continuous: optimal spectral subtraction alpha (0.8 to 3.5)
    y_hpss = []      # Continuous: harmonic energy ratio H / (H + P)
    y_compress = []  # Continuous: optimal compressor gain reduction (dB)
    y_eq = []        # Categorical: target reference EQ shape (0=flat, 1=bright, 2=dark)
    y_rt60 = []      # Continuous: ground truth RT60 decay rate (seconds)
    y_beat = []      # Binary: 1 = frame sits on beat grid, 0 = off beat
    y_timbre = []    # Categorical: instrument source (0=voice, 1=guitar, 2=bell)
    
    for i in range(n_frames):
        # 1. Randomly sample the task configurations with balanced parameters
        f0 = float(rng.uniform(80.0, 600.0))
        gain = float(rng.uniform(0.35, 1.0)) 
        noise_std = float(rng.uniform(0.0, 0.15)) # light to moderate noise
        rt60 = float(rng.uniform(0.1, 1.5))
        eq_shape = int(rng.choice([0, 1, 2])) # 0=flat, 1=bright, 2=dark
        vowel_idx = int(rng.choice([0, 1, 2])) # vowels: /a/, /i/, /u/
        beat_grid = int(rng.choice([0, 1])) # on beat vs off beat
        timbre_class = int(rng.choice([0, 1, 2])) # 0=voice, 1=guitar, 2=bell
        
        # 2. Synthesize base instrument source (timbre)
        sig = np.zeros(N)
        if timbre_class == 0:
            # Voice: formants around 600Hz, 1500Hz + vibrato
            vib = 0.02 * np.sin(2 * np.pi * 6.0 * t)
            phase = 2 * np.pi * f0 * (t + np.cumsum(vib)/sr)
            for h in range(1, 8):
                freq = h * f0
                if freq < sr / 2:
                    # Apply vowel formant envelope
                    if vowel_idx == 0: # /a/
                        formant = np.exp(-((freq - 700)/150)**2) + 0.5 * np.exp(-((freq - 1100)/200)**2)
                    elif vowel_idx == 1: # /i/
                        formant = np.exp(-((freq - 300)/100)**2) + 0.8 * np.exp(-((freq - 2200)/300)**2)
                    else: # /u/
                        formant = np.exp(-((freq - 350)/100)**2) + 0.3 * np.exp(-((freq - 800)/150)**2)
                    sig += (formant + 0.05) * np.sin(h * phase)
        elif timbre_class == 1:
            # Guitar: decaying harmonics
            for h in range(1, 8):
                freq = h * f0
                if freq < sr / 2:
                    sig += (1.0 / h) * np.sin(2 * np.pi * freq * t) * np.exp(-2.0 * h * t)
        else:
            # Bell: inharmonic partials
            partials = [1.0, 1.5, 2.2, 3.1, 4.2]
            for p in partials:
                freq = p * f0
                if freq < sr / 2:
                    sig += np.sin(2 * np.pi * freq * t) * np.exp(-1.5 * t)
                    
        sig = norm01(sig) * gain
        
        # 3. Add transient click for beat grid/onset detection
        percussive = np.zeros(N)
        is_onset = 0
        if beat_grid == 1 or rng.uniform() < 0.25:
            # Click at start of frame
            click_len = int(rng.uniform(0.01, 0.04) * sr)
            tc = np.arange(click_len) / sr
            click = rng.normal(0, 1, click_len) * np.exp(-200 * tc)
            percussive[:click_len] += click
            is_onset = 1
            
        percussive = norm01(percussive) * (gain * rng.uniform(2.5, 6.0))
        
        # Combine harmonic and percussive parts
        mix = sig + percussive
        
        # 4. Apply EQ Shape Filter (mismatch reference test)
        if eq_shape == 1: # Bright filter
            # High-pass filter boost
            mix = scipy.signal.lfilter([1.0, -0.6], [1.0], mix)
        elif eq_shape == 2: # Dark filter
            # Low-pass filter
            mix = scipy.signal.lfilter([1.0], [1.0, -0.85], mix)
            
        mix = norm01(mix) * gain
        
        # 5. Apply Reverb RT60 decay envelope
        tau = rt60 / 2.3026
        decay_env = np.exp(-t / tau)
        mix *= decay_env
        
        # 6. Get clean state reference before noise corrupts it
        st_clean = engine.analyze(mix, sr)
        
        # 7. Add noise
        noisy_mix = mix + rng.normal(0, noise_std, N)
        
        # 8. Analyze noisy frame with engine
        st_noisy = engine.analyze(noisy_mix, sr)
        coords.append(st_noisy.coordinate)
        
        # 9. Record Ground-Truth targets
        harm_e = np.sum(sig ** 2)
        perc_e = np.sum(percussive ** 2)
        
        # Pitch tracking optimal window size: 1 if low f0 or clean signal is noisy, 0 otherwise
        y_pitch.append(1 if (f0 < 130.0 or st_clean.region == "noise_collapse") else 0)
        
        # Voicing detection label: active if clean signal is periodic/harmonic
        y_voicing.append(1 if st_clean.region in ("periodic_harmonic", "smooth_lowpass") else 0)
        
        # Onset detection label: active if clean signal has high transient coordinate z2
        y_onset.append(1 if st_clean.coordinate[1] > 1.2 else 0)
        
        # Optimal denoising alpha: maps directly to the added noise floor
        y_denoising.append(0.8 + 2.2 * (noise_std / 0.15))
        
        # Source separation harmonic ratio: ratio of harmonic to percussive energy in mixture
        y_hpss.append(harm_e / (harm_e + perc_e + 1e-12))
        
        # Optimal compressor gain: proportional to absolute gain (amplitude-blindness test)
        y_compress.append(gain)
        
        # EQ Target Shape (0, 1, 2)
        y_eq.append(eq_shape)
        
        # RT60 value (seconds)
        y_rt60.append(rt60)
        
        # Beat grid alignment
        y_beat.append(beat_grid)
        
        # Timbre class
        y_timbre.append(timbre_class)
        
    coords = np.array(coords)
    targets = {
        "Pitch Tracking":     np.array(y_pitch),
        "Voicing Detection":  np.array(y_voicing),
        "Onset Detection":    np.array(y_onset),
        "Spectral Denoising": np.array(y_denoising),
        "Source Separation":  np.array(y_hpss),
        "Dynamic Compression": np.array(y_compress),
        "EQ Matching":        np.array(y_eq),
        "RT60 Estimation":    np.array(y_rt60),
        "Beat Tracking":      np.array(y_beat),
        "Speaker/Timbre ID":  np.array(y_timbre)
    }
    
    return coords, targets


# ──────────────────────────────────────────────────────────────────────────────
# MODEL EVALUATION
# ──────────────────────────────────────────────────────────────────────────────

def compute_eta_squared(coords, targets):
    """
    Computes eta-squared (variance explained by coordinates) for all 10 tasks.
    """
    eta_scores = {}
    
    regression_tasks = ["Spectral Denoising", "Source Separation", "Dynamic Compression", "RT60 Estimation"]
    classification_tasks = ["Pitch Tracking", "Voicing Detection", "Onset Detection", "EQ Matching", "Beat Tracking", "Speaker/Timbre ID"]
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    for task_name, y in targets.items():
        if task_name in regression_tasks:
            # Fit regressor
            model = RandomForestRegressor(n_estimators=50, max_depth=6, random_state=42)
            scores = cross_val_score(model, coords, y, cv=kf, scoring="r2")
            eta_scores[task_name] = max(0.0, float(np.mean(scores)))
        else:
            # Fit classifier
            model = RandomForestClassifier(n_estimators=50, max_depth=6, random_state=42)
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scores = cross_val_score(model, coords, y, cv=skf, scoring="accuracy")
            avg_acc = float(np.mean(scores))
            
            # Determine chance accuracy
            unique, counts = np.unique(y, return_counts=True)
            chance = np.max(counts) / len(y) # Majority class chance baseline
            
            # Normalize to [0, 1] range representing information explained above chance
            eta = (avg_acc - chance) / (1.0 - chance)
            eta_scores[task_name] = max(0.0, eta)
            
        print(f"Task: {task_name:22s}  η² = {eta_scores[task_name]:.3f}")
        
    return eta_scores


# ──────────────────────────────────────────────────────────────────────────────
# DASHBOARD PLOT
# ──────────────────────────────────────────────────────────────────────────────

def plot_dashboard(eta_scores):
    """
    Plots the final Theory Taxonomy Dashboard.
    """
    # Sort tasks by eta-squared score
    sorted_tasks = sorted(eta_scores.items(), key=lambda x: x[1], reverse=True)
    names = [item[0] for item in sorted_tasks]
    scores = [item[1] for item in sorted_tasks]
    
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(18, 9))
    fig.patch.set_facecolor("#0d1117")
    
    gs = fig.add_gridspec(
        1, 2, wspace=0.35,
        left=0.08, right=0.94, top=0.86, bottom=0.12
    )
    
    BG  = "#161b22"
    TXT = "#c9d1d9"
    ACC = "#f4c542"
    
    # ── Panel 1: Bar Chart ──────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor(BG)
    ax1.set_title("Task State-Space Compatibility Mapping", fontweight="bold", color="white", fontsize=13, pad=12)
    
    # Color mapping: Green to Red
    colors = []
    for s in scores:
        if s >= 0.60:
            colors.append("#52e07f") # Green (highly compatible)
        elif s >= 0.25:
            colors.append("#f4c542") # Gold (partially compatible)
        else:
            colors.append("#e05252") # Red (state-space blind)
            
    bars = ax1.barh(names[::-1], scores[::-1], color=colors[::-1], height=0.6, alpha=0.85)
    ax1.set_xlabel("State Compatibility Index η² (Variance Explained)", color=TXT, fontsize=10, labelpad=8)
    ax1.set_xlim(0, 1.05)
    ax1.grid(axis="x", alpha=0.15, color="#2a2a3a")
    ax1.tick_params(colors=TXT, labelsize=9.5)
    for sp in ax1.spines.values():
        sp.set_color("#30363d")
        
    for bar in bars:
        w_val = bar.get_width()
        ax1.text(w_val + 0.02, bar.get_y() + bar.get_height()/2, f"{w_val:.3f}", 
                 ha="left", va="center", color=TXT, fontsize=9.5, fontweight="bold")
                 
    # ── Panel 2: Summary Taxonomy Table ──────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis("off")
    ax2.set_facecolor(BG)
    ax2.set_title("DSP Theory Taxonomy", fontweight="bold", color="white", fontsize=13, pad=12)
    
    rows = [
        ("Task", "η² Score", "Category", "Compatible?"),
    ]
    for name, s in sorted_tasks:
        if s >= 0.60:
            compat = "✓ Yes (High)"
            cat = "Physical-State"
        elif s >= 0.25:
            compat = "⚠ Partial"
            cat = "State-Influenced"
        else:
            compat = "✗ No (Blind)"
            cat = "External-Domain"
        rows.append((name, f"{s:.3f}", cat, compat))
        
    col_x = [0.02, 0.38, 0.58, 0.82]
    y_start = 0.85
    y_step = 0.08
    for ri, row in enumerate(rows):
        is_header = ri == 0
        for ci, (cell, cx) in enumerate(zip(row, col_x)):
            if is_header:
                color, fw = ACC, "bold"
            elif ci == 3:
                if "Yes" in cell:
                    color = "#52e07f"
                elif "Partial" in cell:
                    color = "#f4c542"
                else:
                    color = "#e05252"
                fw = "bold"
            else:
                color, fw = TXT, "normal"
            ax2.text(cx, y_start - ri * y_step, cell, transform=ax2.transAxes,
                     fontsize=10.5, color=color, fontweight=fw,
                     va="top", ha="left", family="sans-serif")
                     
    fig.suptitle("EXPERIMENT 036 — STATE-SPACE THEORY VALIDATION ATLAS", fontsize=18, fontweight="bold", color="white", y=0.96)
    
    # Save the figure
    os.makedirs(os.path.join(project_root, "results"), exist_ok=True)
    plot_path = os.path.join(project_root, "results/exp036_theory_validation.png")
    plt.savefig(plot_path, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    
    print(f"\nSaved dashboard plot to {plot_path}")
    print("=" * 72)
    print("FINISHED Exp 036 — Theory Validation")
    print("=" * 72)


# ──────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ──────────────────────────────────────────────────────────────────────────────

def run():
    print("=" * 72)
    print("EXPERIMENT 036 — STATE-SPACE THEORY VALIDATION: MEASURING η²")
    print("=" * 72)
    
    engine = RepresentationIntelligenceEngine()
    
    print("Synthesizing validation dataset of 500 audio frames...")
    coords, targets = synthesize_validation_dataset(engine, n_frames=500)
    
    print("\nComputing η² for all 10 tasks...")
    eta_scores = compute_eta_squared(coords, targets)
    
    plot_dashboard(eta_scores)


if __name__ == "__main__":
    run()
