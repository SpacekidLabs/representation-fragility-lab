import sys
import os
import numpy as np
import scipy.signal

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.framework.engine import RepresentationIntelligenceEngine
from src.experiments.exp040_generalization_challenge import norm01

def debug_denoising():
    engine = RepresentationIntelligenceEngine()
    rng = np.random.default_rng(999)
    
    sr = 16000
    # 1.0s tone, 1.5s silence (total 2.5s)
    t = np.arange(int(2.5 * sr)) / sr
    clean = np.zeros(len(t))
    
    clean[:int(1.0*sr)] = np.sin(2 * np.pi * 300.0 * t[:int(1.0*sr)]) + 0.5 * np.sin(2 * np.pi * 600.0 * t[:int(1.0*sr)])
    clean = norm01(clean)
    
    sig_power = np.mean(clean[:int(1.0*sr)]**2)
    noise_power = sig_power / (10 ** (8.0 / 10.0))
    noisy = clean + rng.normal(0, np.sqrt(noise_power), len(clean))
    
    nperseg = 1024
    hop = 256
    noverlap = nperseg - hop
    f, ts, Zxx = scipy.signal.stft(noisy, fs=sr, window="hann", nperseg=nperseg, noverlap=noverlap)
    noise_psd = np.mean(np.abs(Zxx[:, -12:]) ** 2, axis=1) # estimate noise from the silence tail
    
    # Run Static (Baseline)
    Z_bl = np.zeros_like(Zxx, dtype=np.complex128)
    for m in range(Zxx.shape[1]):
        X_mag = np.abs(Zxx[:, m])
        G = np.maximum(1.0 - 2.0 * np.sqrt(noise_psd / (X_mag ** 2 + 1e-12)), 0.05)
        Z_bl[:, m] = Zxx[:, m] * G
    _, den_bl = scipy.signal.istft(Z_bl, fs=sr, window="hann", nperseg=nperseg, noverlap=noverlap)
    den_bl = den_bl[:len(clean)]
    
    # Run Adaptive
    Z_fw = np.zeros_like(Zxx, dtype=np.complex128)
    for m in range(Zxx.shape[1]):
        start = m * hop
        end = start + nperseg
        frame = noisy[start:end]
        if len(frame) < nperseg:
            frame = np.pad(frame, (0, nperseg - len(frame)))
            
        st = engine.analyze(frame, sr)
        if st.region == "noise_collapse":
            alpha = 6.0
            beta = 0.001
        elif st.region in ("periodic_harmonic", "smooth_lowpass"):
            alpha = 1.0
            beta = 0.02
        else:
            alpha = 2.0
            beta = 0.02
            
        X_mag = np.abs(Zxx[:, m])
        G = np.maximum(1.0 - alpha * np.sqrt(noise_psd / (X_mag ** 2 + 1e-12)), beta)
        Z_fw[:, m] = Zxx[:, m] * G
    _, den_fw = scipy.signal.istft(Z_fw, fs=sr, window="hann", nperseg=nperseg, noverlap=noverlap)
    den_fw = den_fw[:len(clean)]
    
    def compute_lsd_local(c, p):
        _, _, Zc = scipy.signal.stft(c, fs=sr, window="hann", nperseg=nperseg, noverlap=noverlap)
        _, _, Zp = scipy.signal.stft(p, fs=sr, window="hann", nperseg=nperseg, noverlap=noverlap)
        mc = 20 * np.log10(np.maximum(np.abs(Zc), 1e-3))
        mp = 20 * np.log10(np.maximum(np.abs(Zp), 1e-3))
        dist = np.sqrt(np.mean((mc - mp) ** 2, axis=0))
        return float(np.mean(dist))
        
    lsd_bl = compute_lsd_local(clean, den_bl)
    lsd_fw = compute_lsd_local(clean, den_fw)
    
    print("LSD Baseline:", lsd_bl)
    print("LSD Adaptive:", lsd_fw)
    
    # Use divisor of 6.0
    p_bl = 1.0 - min(1.0, lsd_bl / 6.0)
    p_fw = 1.0 - min(1.0, lsd_fw / 6.0)
    print("P Baseline:", p_bl)
    print("P Adaptive:", p_fw)
    print("Delta P:", p_fw - p_bl)

if __name__ == "__main__":
    debug_denoising()
