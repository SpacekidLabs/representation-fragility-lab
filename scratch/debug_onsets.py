import sys
import os
import numpy as np
import scipy.signal

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.framework.engine import RepresentationIntelligenceEngine
from src.experiments.exp040_generalization_challenge import norm01, peak_pick, compute_f1_events

def debug_onsets():
    engine = RepresentationIntelligenceEngine()
    rng = np.random.default_rng(999)
    sr = 22050
    notes = [150.0, 250.0, 350.0, 180.0]
    note_dur = 0.3
    gt_onsets = []
    parts = []
    t_cur = 0.0
    for f0 in notes:
        gt_onsets.append(int(t_cur * sr))
        n = int(note_dur * sr)
        t = np.arange(n) / sr
        sig = np.sin(2 * np.pi * f0 * t) + 0.5 * np.sin(2 * np.pi * 2 * f0 * t)
        clk = np.zeros(n)
        clk[:int(0.003 * sr)] = rng.normal(0, 1.5, int(0.003 * sr))
        parts.append((sig + clk) * np.exp(-t * 6.0))
        t_cur += note_dur
    audio = np.concatenate(parts)
    audio = norm01(audio)
    
    # Add false onsets
    for false_t in [0.15, 0.45, 0.75]:
        idx = int(false_t * sr)
        audio[idx : idx + int(0.015 * sr)] += rng.normal(0, 0.15, int(0.015 * sr))
        
    audio = norm01(audio)
    audio += rng.normal(0, 0.02, len(audio))
    audio = norm01(audio)
    
    frame_len = 1024
    hop = 256
    n_frames = (len(audio) - frame_len) // hop
    
    flux = np.zeros(n_frames)
    prev = None
    for i in range(n_frames):
        mag = np.abs(np.fft.rfft(audio[i*hop:i*hop+frame_len] * np.hanning(frame_len)))
        if prev is not None:
            flux[i] = np.sum(np.maximum(mag - prev, 0))
        prev = mag
    flux = norm01(flux)
    
    fw_score = np.zeros(n_frames)
    regions = []
    for i in range(n_frames):
        frame = audio[i*hop:i*hop+frame_len]
        st = engine.analyze(frame, sr)
        regions.append(st.region)
        # Gate based on region
        if st.region == "noise_collapse":
            fw_score[i] = 0.0
        else:
            fw_score[i] = flux[i]
            
    print("Unique regions:", set(regions))
    print("Region counts:", {r: regions.count(r) for r in set(regions)})
    
    min_gap = max(1, int(0.05 * sr / hop))
    
    # Peak pick on gated score
    fw_peaks = peak_pick(fw_score, min_gap, threshold_factor=0.20) * hop
    
    # Peak pick on baseline flux
    bl_peaks = peak_pick(flux, min_gap, threshold_factor=0.20) * hop
    
    print("Detected baseline peaks:", bl_peaks)
    print("Detected adaptive peaks:", fw_peaks)
    print("Ground truth onsets:", gt_onsets)
    
    tol = int(0.05 * sr)
    print("F1 Baseline:", compute_f1_events(bl_peaks, gt_onsets, tol))
    print("F1 Adaptive:", compute_f1_events(fw_peaks, gt_onsets, tol))

if __name__ == "__main__":
    debug_onsets()
