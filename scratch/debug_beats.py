import sys
import os
import numpy as np
import scipy.signal

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.framework.engine import RepresentationIntelligenceEngine
from src.experiments.exp040_generalization_challenge import norm01, peak_pick, compute_f1_events

def debug_beats():
    engine = RepresentationIntelligenceEngine()
    rng = np.random.default_rng(999)
    sr = 22050
    bps = 2.0
    bd = 1.0 / bps
    total_len = int(3 * 4 * bd * sr)
    total = np.zeros(total_len)
    gt_beats = []
    
    def make_kick():
        n = int(0.2 * sr)
        t = np.arange(n) / sr
        f = 70 * np.exp(-t * 20)
        return np.sin(2 * np.pi * np.cumsum(f) / sr) * np.exp(-t * 12)
    def make_snare():
        n = int(0.15 * sr)
        t = np.arange(n) / sr
        sig = rng.normal(0, 1, n)
        b, a = scipy.signal.butter(2, [250/(sr/2), 0.90], btype="band")
        return scipy.signal.filtfilt(b, a, sig) * np.exp(-t * 8)
        
    t_cur = 0.0
    for beat in range(12):
        gt_beats.append(int(t_cur * sr))
        if beat % 2 == 0:
            sig = make_kick()
        else:
            sig = make_snare()
        idx = int(t_cur * sr)
        end = min(idx + len(sig), total_len)
        total[idx:end] += sig[:end - idx]
        t_cur += bd
        
    total = norm01(total)
    
    # Add some random noise bursts in between beats as false alarms
    for false_t in [0.25, 0.75, 1.25, 1.75, 2.25]:
        idx = int(false_t * sr)
        total[idx : idx + int(0.015 * sr)] += rng.normal(0, 0.3, int(0.015 * sr))
        
    total = norm01(total)
    total += rng.normal(0, 0.02, len(total))
    total = norm01(total)
    
    frame_len = 1024
    hop = 256
    n_frames = (len(total) - frame_len) // hop
    
    flux = np.zeros(n_frames)
    prev = None
    for i in range(n_frames):
        mag = np.abs(np.fft.rfft(total[i*hop:i*hop+frame_len] * np.hanning(frame_len)))
        if prev is not None:
            flux[i] = np.sum(np.maximum(mag - prev, 0))
        prev = mag
    flux = norm01(flux)
    min_gap = max(1, int(0.2 * sr / hop))
    
    # Baseline
    bl_peaks = peak_pick(flux, min_gap, threshold_factor=0.25) * hop
    
    # Adaptive
    fw_score = np.zeros(n_frames)
    regions = []
    for i in range(n_frames):
        frame = total[i*hop:i*hop+frame_len]
        st = engine.analyze(frame, sr)
        regions.append(st.region)
        if st.region == "noise_collapse":
            fw_score[i] = flux[i] * 0.1
        else:
            fw_score[i] = flux[i]
            
    fw_score = norm01(fw_score)
    fw_peaks = peak_pick(fw_score, min_gap, threshold_factor=0.25) * hop
    
    print("Region counts:", {r: regions.count(r) for r in set(regions)})
    print("Detected baseline peaks:", bl_peaks)
    print("Detected adaptive peaks:", fw_peaks)
    print("Ground truth beats:", gt_beats)
    
    tol = int(0.08 * sr)
    print("F1 Baseline:", compute_f1_events(bl_peaks, gt_beats, tol))
    print("F1 Adaptive:", compute_f1_events(fw_peaks, gt_beats, tol))

if __name__ == "__main__":
    debug_beats()
