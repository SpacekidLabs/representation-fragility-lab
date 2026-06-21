import sys
import os
import numpy as np
import scipy.signal
import librosa

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.framework.engine import RepresentationIntelligenceEngine
from src.experiments.exp040_generalization_challenge import norm01

def debug_pitch():
    engine = RepresentationIntelligenceEngine()
    rng = np.random.default_rng(999)
    sr = 22050
    dur = 2.0
    n = int(dur * sr)
    t = np.arange(n) / sr
    
    # Ground truth pitch: 120Hz to 270Hz
    f0 = 120.0 + 150.0 * (t / dur)
    phase = 2 * np.pi * np.cumsum(f0) / sr
    clean = np.sin(phase) + 0.5 * np.sin(2 * phase)
    
    noisy = clean.copy()
    
    # Add 5 click bursts
    for click_t in [0.3, 0.6, 0.9, 1.2, 1.5]:
        click_idx = int(click_t * sr)
        noisy[click_idx : click_idx + 1000] += rng.normal(0, 1.5, 1000)
        
    # Noise collapse segment (zero signal, add noise) from 1.6s to 1.9s
    noise_start = int(1.6 * sr)
    noise_end = int(1.9 * sr)
    noisy[noise_start:noise_end] = rng.normal(0, 0.4, noise_end - noise_start)
    
    # Add background noise
    noisy += rng.normal(0, 0.04, n)
    noisy = norm01(noisy)
    
    gt_pitches_full = f0.copy()
    gt_pitches_full[noise_start:noise_end] = 0.0
    
    hop = 512
    pad_len = 2048
    y_pad = np.pad(noisy, pad_len, mode='reflect')
    hop_indices = np.arange(0, n - hop, hop)
    
    bl_pitches = []
    fw_pitches = []
    last_valid_pitch = 120.0
    
    regions = []
    for hop_idx in hop_indices:
        b_win = 2048
        b_frame = y_pad[hop_idx + pad_len - b_win//2 : hop_idx + pad_len + b_win//2]
        try:
            b_p = float(librosa.yin(b_frame, fmin=80, fmax=500, sr=sr, frame_length=b_win, hop_length=b_win, trough_threshold=0.15, center=False)[0])
        except:
            b_p = 0.0
        bl_pitches.append(b_p)
        
        f_win = 1024
        f_frame = y_pad[hop_idx + pad_len - f_win//2 : hop_idx + pad_len + f_win//2]
        st = engine.analyze(f_frame, sr)
        regions.append(st.region)
        win_size = st.recommended_window
        params = st.recommended_parameters["pitch_tracking"]
        trough_thresh = params["yin_trough"]
        
        a_frame = y_pad[hop_idx + pad_len - win_size//2 : hop_idx + pad_len + win_size//2]
        try:
            a_p = float(librosa.yin(a_frame, fmin=80, fmax=500, sr=sr, frame_length=win_size, hop_length=win_size, trough_threshold=trough_thresh, center=False)[0])
        except:
            a_p = 0.0
            
        frame_rms = np.sqrt(np.mean(a_frame**2))
        
        # Optimized threshold
        if frame_rms < 0.12:  # Genuine silence
            a_p = 0.0
        else:
            # Click / noise burst -> hold pitch if ACF safety drops
            if st.region == "transient_overloaded" or st.region == "noise_collapse" or st.assumptions.get("acf", 1.0) < 0.25:
                a_p = last_valid_pitch
            else:
                if a_p > 80.0 and a_p < 500.0:
                    last_valid_pitch = a_p
            
        fw_pitches.append(a_p)
        
    bl_pitches = np.array(bl_pitches)
    fw_pitches = np.array(fw_pitches)
    gt_pitches = gt_pitches_full[hop_indices]
    
    def compute_ger_all(est_pitch, gt_pitch, tolerance=0.20):
        errors = 0
        total = len(gt_pitch)
        for est, gt in zip(est_pitch, gt_pitch):
            if gt == 0.0:
                if est > 0.0:
                    errors += 1
            else:
                if est == 0.0:
                    errors += 1
                elif abs(est - gt) / gt > tolerance:
                    errors += 1
        return errors / total
        
    ger_bl = compute_ger_all(bl_pitches, gt_pitches)
    ger_fw = compute_ger_all(fw_pitches, gt_pitches)
    
    print("GER Baseline:", ger_bl)
    print("GER Adaptive:", ger_fw)
    print("P Baseline:", 1.0 - ger_bl)
    print("P Adaptive:", 1.0 - ger_fw)
    print("Delta P:", ger_bl - ger_fw)

if __name__ == "__main__":
    debug_pitch()
