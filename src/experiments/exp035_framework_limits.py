"""
Experiment 035 — Framework Limits
=================================
Maps exactly where the RepresentationIntelligenceEngine stops working.
We test five tasks where the physical state space is theoretically blind:

  Case A — Source Separation (HPSS)   : mixture state != source composition
  Case B — Dynamic Range Compression  : amplitude-blind state space
  Case C — EQ Matching                : requires target comparison; engine is source-only
  Case D — Reverb / RT60 Estimation   : long-term decay vs instantaneous frame state
  Case E — Timbre / Source ID         : content identity; state space is timbre-blind

Outputs:
  results/exp035_framework_limits.png
"""

import sys
import os
import warnings

import numpy as np
import scipy.signal
import scipy.ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

warnings.filterwarnings("ignore")

from src.framework.engine import RepresentationIntelligenceEngine


# ──────────────────────────────────────────────────────────────────────────────
# SHARED UTILITIES
# ──────────────────────────────────────────────────────────────────────────────

def norm01(arr):
    """Normalise array to [0, 1]."""
    mx = np.max(np.abs(arr))
    return arr / mx if mx > 1e-12 else arr


# ──────────────────────────────────────────────────────────────────────────────
# CASE A: SOURCE SEPARATION (HPSS)
# ──────────────────────────────────────────────────────────────────────────────

def case_a_hpss(engine):
    """
    Separate harmonic and percussive components.
    Baseline: median-filtering HPSS.
    Framework: uses acf_safety as a frame-level trust gate to weight the masks.
    """
    sr = 16000
    dur = 2.0
    n_samples = int(dur * sr)
    t = np.arange(n_samples) / sr
    
    # Sweep harmonic signal (200 -> 600 Hz)
    f_sweep = 200 + 400 * (t / dur)
    phase = 2 * np.pi * np.cumsum(f_sweep) / sr
    harmonic = np.sin(phase) + 0.5 * np.sin(2 * phase) + 0.25 * np.sin(3 * phase)
    harmonic = norm01(harmonic) * 0.7
    
    # Percussive clicks (every 0.4 seconds)
    percussive = np.zeros(n_samples)
    click_positions = [int(0.2 * sr), int(0.6 * sr), int(1.0 * sr), int(1.4 * sr), int(1.8 * sr)]
    rng = np.random.default_rng(42)
    for pos in click_positions:
        click_len = int(0.05 * sr)
        tc = np.arange(click_len) / sr
        click = rng.normal(0, 1, click_len) * np.exp(-150 * tc)
        percussive[pos:pos+click_len] += click
    percussive = norm01(percussive) * 0.5
    
    mixture = harmonic + percussive
    
    # STFT parameters
    nperseg = 1024
    noverlap = 512
    f, ts, Zxx = scipy.signal.stft(mixture, fs=sr, nperseg=nperseg, noverlap=noverlap)
    mag = np.abs(Zxx)
    phase = np.angle(Zxx)
    
    # Baseline Fitzgerald HPSS
    H_mag = scipy.ndimage.median_filter(mag, size=(1, 15))
    P_mag = scipy.ndimage.median_filter(mag, size=(15, 1))
    
    mask_h = H_mag / (H_mag + P_mag + 1e-12)
    
    # Invert baseline
    Zxx_h_bl = (mag * mask_h) * np.exp(1j * phase)
    _, h_bl = scipy.signal.istft(Zxx_h_bl, fs=sr, nperseg=nperseg, noverlap=noverlap)
    h_bl = h_bl[:n_samples]
    
    # Framework-assisted: frame-level ACF safety scaling
    w_acf = np.zeros(len(ts))
    for k in range(len(ts)):
        sample_idx = k * noverlap
        frame = mixture[sample_idx:sample_idx+nperseg]
        if len(frame) < nperseg:
            frame = np.pad(frame, (0, nperseg - len(frame)))
        st = engine.analyze(frame, sr)
        w_acf[k] = st.assumptions["acf"]
        
    # Scale mask by ACF safety
    mask_h_fw = mask_h * w_acf[np.newaxis, :]
    Zxx_h_fw = (mag * mask_h_fw) * np.exp(1j * phase)
    _, h_fw = scipy.signal.istft(Zxx_h_fw, fs=sr, nperseg=nperseg, noverlap=noverlap)
    h_fw = h_fw[:n_samples]
    
    # SDR Metric: target is the clean harmonic component
    def sdr(ref, est):
        err = np.mean((ref - est) ** 2)
        ref_power = np.mean(ref ** 2)
        return 10 * np.log10(ref_power / (err + 1e-12))
        
    sdr_bl = sdr(harmonic, h_bl)
    sdr_fw = sdr(harmonic, h_fw)
    
    print(f"Case A (HPSS) SDR (Harmonic Recovery):")
    print(f"  Baseline Fitzgerald: {sdr_bl:.2f} dB")
    print(f"  Framework-assisted : {sdr_fw:.2f} dB")
    print(f"  Delta              : {sdr_fw - sdr_bl:+.2f} dB")
    
    return sdr_bl, sdr_fw


# ──────────────────────────────────────────────────────────────────────────────
# CASE B: DYNAMIC RANGE COMPRESSION
# ──────────────────────────────────────────────────────────────────────────────

def case_b_compressor(engine):
    """
    Compress dynamic range of sweep with quiet, loud, quiet sections.
    Baseline: Standard peak feedforward compressor.
    Framework: Adapts threshold depending on region.
    """
    sr = 16000
    dur_sec = 3.0
    n_samples = int(dur_sec * sr)
    t = np.arange(n_samples) / sr
    
    # Sine wave with three amplitude segments: quiet (0.05) -> loud (0.8) -> quiet (0.05)
    amp_env = np.ones(n_samples) * 0.05
    amp_env[int(1.0 * sr):int(2.0 * sr)] = 0.8
    
    # Smooth boundaries slightly to avoid clicks
    b, a = scipy.signal.butter(2, 20 / (sr/2), btype="low")
    amp_env = scipy.signal.filtfilt(b, a, amp_env)
    
    signal = amp_env * np.sin(2 * np.pi * 440 * t) + np.random.default_rng(0).normal(0, 0.001, n_samples)
    
    # Compressor parameters
    threshold_db = -20.0
    ratio = 4.0
    attack_ms = 10.0
    release_ms = 100.0
    
    alpha_a = np.exp(-1.0 / (attack_ms * 1e-3 * sr))
    alpha_r = np.exp(-1.0 / (release_ms * 1e-3 * sr))
    
    # Baseline Peak-detecting Compressor
    def run_compressor(x, thresh_seq):
        y = 0.0
        g = np.ones(len(x))
        for n in range(len(x)):
            abs_x = abs(x[n])
            if abs_x > y:
                y = alpha_a * y + (1.0 - alpha_a) * abs_x
            else:
                y = alpha_r * y + (1.0 - alpha_r) * abs_x
            
            y_db = 20 * np.log10(y + 1e-9)
            thresh = thresh_seq[n]
            if y_db > thresh:
                g_db = - (y_db - thresh) * (1.0 - 1.0 / ratio)
            else:
                g_db = 0.0
            g[n] = 10 ** (g_db / 20.0)
        return x * g
        
    # Baseline threshold sequence (constant)
    thresh_bl = np.ones(n_samples) * threshold_db
    compressed_bl = run_compressor(signal, thresh_bl)
    
    # Framework-assisted threshold sequence:
    # Query engine per 512 samples to adapt threshold.
    hop = 512
    n_frames = n_samples // hop
    thresh_fw = np.ones(n_samples) * threshold_db
    
    for i in range(n_frames):
        frame = signal[i*hop:(i+1)*hop]
        st = engine.analyze(frame, sr)
        if st.region == "noise_collapse":
            t_val = threshold_db + 10.0
        elif st.region == "periodic_harmonic":
            t_val = threshold_db - 5.0
        else:
            t_val = threshold_db
        thresh_fw[i*hop:(i+1)*hop] = t_val
        
    compressed_fw = run_compressor(signal, thresh_fw)
    
    # Metric: Ratio of RMS in the loud segment to RMS in the quiet segments.
    # A smaller ratio means better dynamic range compression.
    def compute_dr(sig):
        rms_quiet1 = np.sqrt(np.mean(sig[int(0.2*sr):int(0.8*sr)]**2))
        rms_loud   = np.sqrt(np.mean(sig[int(1.2*sr):int(1.8*sr)]**2))
        rms_quiet2 = np.sqrt(np.mean(sig[int(2.2*sr):int(2.8*sr)]**2))
        avg_quiet = 0.5 * (rms_quiet1 + rms_quiet2)
        return 20 * np.log10(rms_loud / (avg_quiet + 1e-12))
        
    dr_orig = compute_dr(signal)
    dr_bl = compute_dr(compressed_bl)
    dr_fw = compute_dr(compressed_fw)
    
    print(f"Case B (Compression) Dynamic Range Ratio:")
    print(f"  Unprocessed        : {dr_orig:.2f} dB")
    print(f"  Baseline Compressor: {dr_bl:.2f} dB")
    print(f"  Framework-assisted : {dr_fw:.2f} dB")
    print(f"  Delta              : {dr_fw - dr_bl:+.2f} dB")
    
    return dr_bl, dr_fw, dr_orig


# ──────────────────────────────────────────────────────────────────────────────
# CASE C: EQ MATCHING
# ──────────────────────────────────────────────────────────────────────────────

def case_c_eq_matching(engine):
    """
    Match source spectrum to a shaped target spectrum.
    Baseline: spectral ratio matching with fixed gaussian smoothing.
    Framework: adapts gaussian smoothing width based on source region.
    """
    sr = 16000
    dur = 2.0
    n_samples = int(dur * sr)
    t = np.arange(n_samples) / sr
    
    # Source: flat-spectrum harmonic signal
    f0 = 150.0
    source = np.zeros(n_samples)
    for h in range(1, 25):
        source += np.sin(2 * np.pi * h * f0 * t) / h
    source = norm01(source) * 0.5
    
    # Target: boost at mid frequencies + tilt (+6dB at 1.5kHz)
    b, a = scipy.signal.butter(4, [800 / (sr/2), 2200 / (sr/2)], btype="bandpass")
    mid_boost = scipy.signal.filtfilt(b, a, source) * 3.0
    target = source + mid_boost
    target = norm01(target) * 0.5
    
    fft_len = 2048
    hop = 512
    win = np.hanning(fft_len)
    
    # Baseline EQ matching
    def compute_average_spectrum(sig):
        n_frames = (len(sig) - fft_len) // hop
        specs = []
        for i in range(n_frames):
            frame = sig[i*hop:i*hop+fft_len] * win
            specs.append(np.abs(np.fft.rfft(frame)))
        return np.mean(specs, axis=0) + 1e-9
        
    avg_src = compute_average_spectrum(source)
    avg_tgt = compute_average_spectrum(target)
    ratio = avg_tgt / avg_src
    
    # Baseline: fixed smoothing (15 bins)
    ratio_bl = scipy.ndimage.gaussian_filter1d(ratio, 15)
    
    # Apply baseline filter to source
    def apply_filter(sig, filt):
        n_frames = (len(sig) - fft_len) // hop
        out = np.zeros_like(sig)
        norm_v = np.zeros_like(sig)
        for i in range(n_frames):
            frame = sig[i*hop:i*hop+fft_len]
            X = np.fft.rfft(frame * win)
            X_out = X * filt
            rec = np.fft.irfft(X_out) * win
            out[i*hop:i*hop+fft_len] += rec
            norm_v[i*hop:i*hop+fft_len] += win ** 2
        norm_v = np.maximum(norm_v, 1e-9)
        return out / norm_v
        
    eq_bl = apply_filter(source, ratio_bl)
    
    # Framework attempt: adapt smoothing bandwidth based on source region per frame.
    # Note: since the source is static periodic_harmonic, region is constant.
    n_frames = (len(source) - fft_len) // hop
    eq_fw = np.zeros_like(source)
    norm_v_fw = np.zeros_like(source)
    
    for i in range(n_frames):
        frame = source[i*hop:i*hop+fft_len]
        st = engine.analyze(frame, sr)
        if st.region == "noise_collapse":
            smooth_bins = 40
        elif st.region == "periodic_harmonic":
            smooth_bins = 5
        else:
            smooth_bins = 15
            
        # compute frame ratio
        frame_mag = np.abs(np.fft.rfft(frame * win)) + 1e-9
        frame_ratio = avg_tgt / frame_mag
        frame_ratio_smooth = scipy.ndimage.gaussian_filter1d(frame_ratio, smooth_bins)
        
        X = np.fft.rfft(frame * win)
        X_out = X * frame_ratio_smooth
        rec = np.fft.irfft(X_out) * win
        eq_fw[i*hop:i*hop+fft_len] += rec
        norm_v_fw[i*hop:i*hop+fft_len] += win ** 2
        
    norm_v_fw = np.maximum(norm_v_fw, 1e-9)
    eq_fw = eq_fw / norm_v_fw
    
    # Metric: Log Spectral Distance (LSD) in dB to target
    def lsd(ref, est):
        n_frames = (len(ref) - fft_len) // hop
        lsd_vals = []
        for i in range(n_frames):
            ref_mag = np.abs(np.fft.rfft(ref[i*hop:i*hop+fft_len] * win)) + 1e-9
            est_mag = np.abs(np.fft.rfft(est[i*hop:i*hop+fft_len] * win)) + 1e-9
            ref_db = 20 * np.log10(ref_mag)
            est_db = 20 * np.log10(est_mag)
            lsd_vals.append(np.sqrt(np.mean((ref_db - est_db) ** 2)))
        return float(np.mean(lsd_vals))
        
    orig_lsd = lsd(target, source)
    lsd_bl = lsd(target, eq_bl)
    lsd_fw = lsd(target, eq_fw)
    
    print(f"Case C (EQ Matching) LSD to Target:")
    print(f"  Unprocessed        : {orig_lsd:.2f} dB")
    print(f"  Baseline Matching  : {lsd_bl:.2f} dB")
    print(f"  Framework-assisted : {lsd_fw:.2f} dB")
    print(f"  Delta              : {lsd_fw - lsd_bl:+.2f} dB")
    
    return lsd_bl, lsd_fw, orig_lsd


# ──────────────────────────────────────────────────────────────────────────────
# CASE D: REVERB / RT60 ESTIMATION
# ──────────────────────────────────────────────────────────────────────────────

def case_d_rt60(engine):
    """
    Estimate RT60 from decaying sine wave.
    Baseline: Schroeder backward integration.
    Framework: tracks acf_safety drop down to threshold.
    """
    sr = 16000
    target_rt60s = [0.15, 0.45, 0.75, 1.20]
    
    errors_bl = []
    errors_fw = []
    
    for T in target_rt60s:
        # Generate decay curve: sine wave with exponential decay
        tau = T / 2.3026
        # Let's run for 2.5 seconds
        dur = 2.5
        n_samples = int(dur * sr)
        t = np.arange(n_samples) / sr
        
        # Cutoff at 0.3s
        cut_off = int(0.3 * sr)
        signal = np.zeros(n_samples)
        
        # Play dry for 0.3s
        signal[:cut_off] = np.sin(2 * np.pi * 440.0 * t[:cut_off])
        
        # Decay after cutoff
        t_decay = t[cut_off:] - t[cut_off]
        signal[cut_off:] = np.sin(2 * np.pi * 440.0 * t[cut_off:]) * np.exp(-t_decay / tau)
        
        # Add background noise (-60 dBFS)
        signal += np.random.default_rng(42).normal(0, 1e-3, n_samples)
        signal = norm01(signal)
        
        # 1. Schroeder baseline
        decay_sig = signal[cut_off:]
        edc = np.flip(np.cumsum(np.flip(decay_sig ** 2)))
        edc_db = 10 * np.log10(edc / (edc[0] + 1e-12) + 1e-12)
        
        # Fit a line between -5 dB and -35 dB
        idx_5 = np.where(edc_db <= -5.0)[0]
        idx_35 = np.where(edc_db <= -35.0)[0]
        
        if len(idx_5) > 0 and len(idx_35) > 0:
            i_start = idx_5[0]
            i_end = idx_35[0]
            if i_end > i_start + 100:
                x_fit = np.arange(i_start, i_end)
                y_fit = edc_db[i_start:i_end]
                slope, intercept = np.polyfit(x_fit, y_fit, 1)
                # RT60 is the time to decay by 60 dB
                est_rt60_bl = -60.0 / (slope * sr)
            else:
                est_rt60_bl = 0.0
        else:
            est_rt60_bl = 0.0
            
        # 2. Framework: Track acf_safety
        hop = 256
        frame_len = 1024
        n_frames = (len(signal) - frame_len) // hop
        acf_safeties = []
        for i in range(n_frames):
            frame = signal[i*hop:i*hop+frame_len]
            st = engine.analyze(frame, sr)
            acf_safeties.append(st.assumptions["acf"])
            
        start_frame = cut_off // hop
        end_frame = start_frame
        for i in range(start_frame, len(acf_safeties)):
            if acf_safeties[i] < 0.5:
                end_frame = i
                break
        if end_frame == start_frame:
            end_frame = len(acf_safeties) - 1
            
        est_rt60_fw = (end_frame - start_frame) * hop / sr
        
        errors_bl.append(abs(est_rt60_bl - T))
        errors_fw.append(abs(est_rt60_fw - T))
        
    mae_bl = float(np.mean(errors_bl))
    mae_fw = float(np.mean(errors_fw))
    
    print(f"Case D (RT60 Estimation) MAE Error:")
    print(f"  Schroeder Baseline : {mae_bl:.3f} s")
    print(f"  Framework-assisted : {mae_fw:.3f} s")
    print(f"  Delta              : {mae_fw - mae_bl:+.3f} s")
    
    return mae_bl, mae_fw, target_rt60s, errors_bl, errors_fw


# ──────────────────────────────────────────────────────────────────────────────
# CASE E: TIMBRE / SOURCE IDENTIFICATION
# ──────────────────────────────────────────────────────────────────────────────

def case_e_timbre(engine):
    """
    Classify 3 sources (voice, guitar, bell) at same f0=220Hz.
    Baseline: Timbre features (spectral centroid, spread, kurtosis, 4-bin DCT).
    Framework: Physical state space features (z1, z2, stft_s, acf_s, cep_s).
    """
    sr = 16000
    dur = 0.5
    n_samples = int(dur * sr)
    t = np.arange(n_samples) / sr
    
    # Synthesize isolated notes
    def make_voice(f0, index):
        # Voice: harmonic series with formants around 600Hz and 1600Hz + vibrato
        rng = np.random.default_rng(index)
        vib = 0.02 * np.sin(2 * np.pi * (5.5 + rng.normal(0, 0.5)) * t)
        phase = 2 * np.pi * f0 * (t + np.cumsum(vib)/sr)
        sig = np.zeros(n_samples)
        for h in range(1, 10):
            freq = h * f0
            # formants
            formant1 = np.exp(-((freq - 600) / 150)**2)
            formant2 = np.exp(-((freq - 1600) / 300)**2)
            amp = formant1 + 0.3 * formant2 + 0.05
            sig += amp * np.sin(h * phase)
        # short attack/release env
        env = np.ones(n_samples)
        env[:int(0.02*sr)] = np.linspace(0, 1, int(0.02*sr))
        env[-int(0.05*sr):] = np.linspace(1, 0, int(0.05*sr))
        return norm01(sig * env)
        
    def make_guitar(f0, index):
        # Guitar: decaying harmonics (higher harmonics decay faster)
        rng = np.random.default_rng(index + 100)
        sig = np.zeros(n_samples)
        for h in range(1, 10):
            decay_rate = 3.0 * h + rng.normal(0, 0.5)
            sig += (1.0 / h) * np.sin(2 * np.pi * h * f0 * t) * np.exp(-decay_rate * t)
        # pluck transient env
        env = np.ones(n_samples)
        env[:int(0.005*sr)] = np.linspace(0, 1, int(0.005*sr))
        env[-int(0.03*sr):] = np.linspace(1, 0, int(0.03*sr))
        return norm01(sig * env)
        
    def make_bell(f0, index):
        # Bell: fundamental + inharmonics with slow decay
        rng = np.random.default_rng(index + 200)
        sig = np.zeros(n_samples)
        partials = [1.0, 1.52, 2.21, 3.12, 4.25]
        for p in partials:
            decay_rate = 1.5 + rng.normal(0, 0.2)
            sig += np.sin(2 * np.pi * p * f0 * t) * np.exp(-decay_rate * t)
        env = np.ones(n_samples)
        env[:int(0.002*sr)] = np.linspace(0, 1, int(0.002*sr))
        env[-int(0.03*sr):] = np.linspace(1, 0, int(0.03*sr))
        return norm01(sig * env)
        
    # Generate 15 samples per class (total 45)
    dataset = []
    labels = []
    classes = ["voice", "guitar", "bell"]
    
    for i in range(15):
        dataset.append((make_voice(220.0, i), 0))
        dataset.append((make_guitar(220.0, i), 1))
        dataset.append((make_bell(220.0, i), 2))
        
    # Shuffle dataset
    rng_shuffle = np.random.default_rng(77)
    indices = np.arange(len(dataset))
    rng_shuffle.shuffle(indices)
    
    # Standard split: 30 train, 15 test
    train_idx = indices[:30]
    test_idx = indices[30:]
    
    # Feature extraction
    def extract_baseline_features(x):
        # Spectral features: centroid, spread, kurtosis + 4-bin DCT of log spectrum
        win = np.hanning(len(x))
        X = np.abs(np.fft.rfft(x * win)) + 1e-9
        freqs = np.fft.rfftfreq(len(x), 1.0/sr)
        
        centroid = np.sum(freqs * X) / np.sum(X)
        spread = np.sqrt(np.sum(((freqs - centroid)**2) * X) / np.sum(X))
        
        mean_X = np.mean(X)
        std_X = np.std(X) + 1e-12
        kurtosis = np.mean(((X - mean_X) / std_X)**4)
        
        log_spec = np.log10(X)
        dct = scipy.fftpack.dct(log_spec, type=2, norm="ortho")[:4]
        
        return np.array([centroid, spread, kurtosis, dct[0], dct[1], dct[2], dct[3]])
        
    def extract_framework_features(x):
        # We run the engine on the middle frame
        mid_frame = x[len(x)//4 : len(x)//4 + 1024]
        if len(mid_frame) < 1024:
            mid_frame = np.pad(mid_frame, (0, 1024 - len(mid_frame)))
        st = engine.analyze(mid_frame, sr)
        return np.array([
            st.coordinate[0], st.coordinate[1],
            st.assumptions["stft"], st.assumptions["acf"], st.assumptions["cepstrum"]
        ])
        
    # Baseline Feature Matrix
    feat_bl = np.array([extract_baseline_features(item[0]) for item in dataset])
    # Standardize baseline features
    bl_mean = np.mean(feat_bl, axis=0)
    bl_std = np.std(feat_bl, axis=0) + 1e-12
    feat_bl_std = (feat_bl - bl_mean) / bl_std
    
    # Framework Feature Matrix
    feat_fw = np.array([extract_framework_features(item[0]) for item in dataset])
    fw_mean = np.mean(feat_fw, axis=0)
    fw_std = np.std(feat_fw, axis=0) + 1e-12
    feat_fw_std = (feat_fw - fw_mean) / fw_std
    
    # Train nearest centroid classifier
    def train_and_eval(feats):
        train_feats = feats[train_idx]
        train_labels = np.array([dataset[idx][1] for idx in train_idx])
        test_feats = feats[test_idx]
        test_labels = np.array([dataset[idx][1] for idx in test_idx])
        
        # Compute centroids for each class
        centroids = []
        for c in range(3):
            centroids.append(np.mean(train_feats[train_labels == c], axis=0))
        centroids = np.array(centroids)
        
        # Predict on test
        correct = 0
        for i in range(len(test_feats)):
            dists = np.sum((centroids - test_feats[i])**2, axis=1)
            pred = np.argmin(dists)
            if pred == test_labels[i]:
                correct += 1
        return correct / len(test_feats)
        
    acc_bl = train_and_eval(feat_bl_std)
    acc_fw = train_and_eval(feat_fw_std)
    
    print(f"Case E (Timbre Classification) Test Accuracy (Chance = 33.3%):")
    print(f"  Baseline features  : {acc_bl * 100.0:.1f}%")
    print(f"  Framework features : {acc_fw * 100.0:.1f}%")
    print(f"  Delta              : {(acc_fw - acc_bl) * 100.0:+.1f}%")
    
    return acc_bl, acc_fw


# ──────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE + PLOT
# ──────────────────────────────────────────────────────────────────────────────

def run():
    print("=" * 72)
    print("EXPERIMENT 035 — FRAMEWORK LIMITS: WHERE DOES THE ENGINE STOP WORKING?")
    print("=" * 72)
    
    engine = RepresentationIntelligenceEngine()
    
    # 1. Case A: HPSS
    sdr_bl, sdr_fw = case_a_hpss(engine)
    
    # 2. Case B: Compression
    dr_bl, dr_fw, dr_orig = case_b_compressor(engine)
    
    # 3. Case C: EQ Matching
    lsd_bl, lsd_fw, orig_lsd = case_c_eq_matching(engine)
    
    # 4. Case D: Reverb RT60
    mae_bl, mae_fw, rt60_targets, errors_bl, errors_fw = case_d_rt60(engine)
    
    # 5. Case E: Timbre Classification
    acc_bl, acc_fw = case_e_timbre(engine)
    
    # ──────────────────────────────────────────────────────────────────────────
    # PLOT: dark dashboard
    # ──────────────────────────────────────────────────────────────────────────
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(22, 12))
    fig.patch.set_facecolor("#0d1117")
    
    gs = fig.add_gridspec(
        2, 3, hspace=0.45, wspace=0.35,
        left=0.06, right=0.96, top=0.88, bottom=0.08
    )
    
    BL  = "#e05252"   # baseline red
    AS  = "#52b0e0"   # assisted blue
    ACC = "#f4c542"   # accent gold
    TXT = "#c9d1d9"
    GRD = "#2a2a3a"
    BG  = "#161b22"
    
    def style_ax(ax, title):
        ax.set_facecolor(BG)
        ax.set_title(title, fontweight="bold", color="white", fontsize=12, pad=10)
        ax.tick_params(colors=TXT, labelsize=9)
        ax.grid(axis="y", alpha=0.15, color=GRD)
        for sp in ax.spines.values():
            sp.set_color("#30363d")
            
    # ── Panel 1: Case A (HPSS) ──────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    bars1 = ax1.bar(["Baseline Fitzgerald", "Framework-Assisted"], [sdr_bl, sdr_fw], color=[BL, AS], width=0.45, alpha=0.85)
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, yval + 0.3, f"{yval:.2f} dB", ha="center", va="bottom", color=TXT, fontsize=10, fontweight="bold")
    ax1.set_ylabel("SDR (dB)", color=TXT, fontsize=10)
    ax1.set_ylim(0, max(sdr_bl, sdr_fw) + 4)
    style_ax(ax1, "Case A: HPSS (Harmonic Recovery SDR)")
    
    # ── Panel 2: Case B (Compression) ───────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    bars2 = ax2.bar(["Unprocessed", "Baseline Peak", "Framework-Assisted"], [dr_orig, dr_bl, dr_fw], color=["#888", BL, AS], width=0.45, alpha=0.85)
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, yval + 0.5, f"{yval:.1f} dB", ha="center", va="bottom", color=TXT, fontsize=10, fontweight="bold")
    ax2.set_ylabel("Loud-to-Quiet Ratio (dB)", color=TXT, fontsize=10)
    ax2.set_ylim(0, dr_orig + 4)
    style_ax(ax2, "Case B: Dynamic Range Compression")
    
    # ── Panel 3: Case C (EQ Matching) ───────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    bars3 = ax3.bar(["Unprocessed", "Baseline Matching", "Framework-Assisted"], [orig_lsd, lsd_bl, lsd_fw], color=["#888", BL, AS], width=0.45, alpha=0.85)
    for bar in bars3:
        yval = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2, yval + 0.3, f"{yval:.2f} dB", ha="center", va="bottom", color=TXT, fontsize=10, fontweight="bold")
    ax3.set_ylabel("LSD to Target (dB)", color=TXT, fontsize=10)
    ax3.set_ylim(0, orig_lsd + 4)
    style_ax(ax3, "Case C: EQ Matching Error")
    
    # ── Panel 4: Case D (RT60 Estimation) ───────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    w = 0.35
    x = np.arange(len(rt60_targets))
    ax4.bar(x - w/2, errors_bl, w, color=BL, alpha=0.85, label="Schroeder (Baseline)")
    ax4.bar(x + w/2, errors_fw, w, color=AS, alpha=0.85, label="ACF-Drop (Framework)")
    ax4.set_xticks(x)
    ax4.set_xticklabels([f"{t}s" for t in rt60_targets])
    ax4.set_xlabel("Target RT60", color=TXT, fontsize=10)
    ax4.set_ylabel("Absolute Error (seconds)", color=TXT, fontsize=10)
    ax4.legend(fontsize=8, loc="upper left")
    style_ax(ax4, "Case D: RT60 Estimation Error")
    
    # ── Panel 5: Case E (Timbre Classification) ─────────────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    bars5 = ax5.bar(["Baseline Features", "Framework Features"], [acc_bl * 100, acc_fw * 100], color=[BL, AS], width=0.45, alpha=0.85)
    for bar in bars5:
        yval = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2, yval + 2.0, f"{yval:.1f}%", ha="center", va="bottom", color=TXT, fontsize=10, fontweight="bold")
    # Draw chance line (33.3%)
    ax5.axhline(33.3, color="#ff4444", linestyle="--", alpha=0.7, label="Chance (33.3%)")
    ax5.set_ylabel("Classification Accuracy (%)", color=TXT, fontsize=10)
    ax5.set_ylim(0, 110)
    ax5.legend(fontsize=8, loc="upper right")
    style_ax(ax5, "Case E: Timbre Classification")
    
    # ── Panel 6: Summary Limits Table ───────────────────────────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis("off")
    ax6.set_facecolor(BG)
    ax6.set_title("Framework Limits Summary", fontweight="bold", color="white", fontsize=12, pad=10)
    
    rows = [
        ("Task", "Reason engine is blind", "Works?"),
        ("Source Separation", "Mixture state != source composition", "✗ No"),
        ("Dynamic Compression", "Amplitude-domain; state amp-blind", "✗ No"),
        ("EQ Matching", "Requires target; engine source-only", "✗ No"),
        ("RT60 Estimation", "Long-term temporal; engine frame-level", "✗ No"),
        ("Timbre ID", "Content identity; state timbre-blind", "✗ No"),
    ]
    
    col_x = [0.02, 0.36, 0.85]
    y_start = 0.85
    y_step = 0.12
    for ri, row in enumerate(rows):
        is_header = ri == 0
        for ci, (cell, cx) in enumerate(zip(row, col_x)):
            if is_header:
                color, fw = ACC, "bold"
            elif ci == 2:
                color = "#ff5555"
                fw = "bold"
            else:
                color, fw = TXT, "normal"
            ax6.text(cx, y_start - ri * y_step, cell, transform=ax6.transAxes,
                     fontsize=9.5, color=color, fontweight=fw,
                     va="top", ha="left", family="sans-serif")
                     
    # Add title text
    fig.suptitle("EXPERIMENT 035 — FRAMEWORK LIMITS ATLAS", fontsize=18, fontweight="bold", color="white", y=0.96)
    
    # Save the figure
    os.makedirs(os.path.join(project_root, "results"), exist_ok=True)
    plot_path = os.path.join(project_root, "results/exp035_framework_limits.png")
    plt.savefig(plot_path, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    
    print(f"\nSaved dashboard plot to {plot_path}")
    print("=" * 72)
    print("FINISHED Exp 035 — Framework Limits")
    print("=" * 72)


if __name__ == "__main__":
    run()
