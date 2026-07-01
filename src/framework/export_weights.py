"""
Export Weights Script
=====================
Runs the calibration pipeline on the 10 signal classes and 4,000 frames from Exp 031,
fits the PCA projection and the 5 polynomial Ridge Regression boundaries, and exports
them directly to src/framework/weights.py as constant numpy arrays.
"""

import sys
import os
import warnings
import numpy as np
import scipy.signal
import librosa

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

warnings.filterwarnings('ignore', category=UserWarning)

from src.representations.acf import compute_acf
from src.representations.cepstrum import compute_cepstrum
from src.representations.stft import compute_stft
from src.framework.descriptors import extract_physical_descriptors
from src.experiments.exp028_failure_manifold_validation import (
    compute_cqt_frame,
    compute_morlet_cwt,
    estimate_pitch_stft,
    estimate_pitch_acf,
    estimate_pitch_cepstrum,
    estimate_pitch_cqt,
    synthesize_karplus_strong,
    extract_high_energy_frames,
    apply_random_perturbation,
    cosine_similarity
)
from src.experiments.exp030_universal_audio_state_space import (
    synthesize_synth_waves,
    synthesize_fm_waves,
    synthesize_granular_textures,
    generate_noise_frames,
)
from src.experiments.exp031_assumption_surfaces import (
    estimate_pitch_cwt,
    get_poly_features,
    train_ridge_regression
)

def run():
    print("=" * 75)
    print("FRAMEWORK API — TRAINING WEIGHTS EXPORTER")
    print("=" * 75)
    sr = 22050
    win = 1024
    N = 200
    N_total = N * 10
    rng = np.random.default_rng(42)

    print("\n1. Loading and synthesizing calibration signals (10 classes)...")
    
    # Load identical calibration files
    y_sp, _ = librosa.load(librosa.example('libri1'), sr=sr)
    speech_frames = extract_high_energy_frames(y_sp, win, N)

    y_voc, _ = librosa.load(os.path.join(project_root, "Clean_vocal.wav"), sr=sr)
    vocals_frames = extract_high_energy_frames(y_voc, win, N)

    y_pn, _ = librosa.load(librosa.example('pistachio'), sr=sr)
    piano_frames = extract_high_energy_frames(y_pn, win, N)

    y_dr, _ = librosa.load(librosa.example('choice'), sr=sr)
    drums_frames = extract_high_energy_frames(y_dr, win, N)

    guitar_frames = []
    f0s = [82.4, 110.0, 146.8, 196.0, 220.0, 329.6]
    while len(guitar_frames) < N:
        f0 = rng.choice(f0s)
        pluck = synthesize_karplus_strong(f0, sr, int(1.2 * sr))
        guitar_frames.extend(extract_high_energy_frames(pluck, win, 20))
    guitar_frames = guitar_frames[:N]

    y_rb, _ = librosa.load(librosa.example('robin'), sr=sr)
    robin_frames = extract_high_energy_frames(y_rb, win, N)

    synth_frames = []
    while len(synth_frames) < N:
        synth_frames.extend(synthesize_synth_waves(rng.choice(f0s), sr, win, 50))
    synth_frames = synth_frames[:N]

    fm_frames = []
    while len(fm_frames) < N:
        fm_frames.extend(synthesize_fm_waves(rng.choice(f0s), sr, win, 50))
    fm_frames = fm_frames[:N]

    granular_frames = []
    while len(granular_textures := synthesize_granular_textures(sr, win, N)) < N:
        pass
    granular_frames = granular_textures[:N]

    noise_frames = generate_noise_frames(win, N)

    all_corpora = {
        "Speech": speech_frames,
        "Vocals": vocals_frames,
        "Piano": piano_frames,
        "Drums": drums_frames,
        "Guitar": guitar_frames,
        "Environmental": robin_frames,
        "Synths": synth_frames,
        "FM Tone": fm_frames,
        "Granular": granular_frames,
        "Noise": noise_frames
    }

    print("\n2. Extracting physical features for clean reference frames...")
    clean_representations = []
    X_clean_feats = []
    corpus_labels = []

    for c_name, frames in all_corpora.items():
        for frame in frames:
            acf = compute_acf(frame)
            cep = compute_cepstrum(frame)
            mag = compute_stft(frame, sr)
            if mag.ndim > 1: mag = np.mean(mag, axis=1)
            cqt = compute_cqt_frame(frame, sr)
            cwt = compute_morlet_cwt(frame, 64)
            clean_representations.append((acf, cep, mag, cqt, cwt))
            
            feats = extract_physical_descriptors(frame, sr)
            X_clean_feats.append(feats)
            corpus_labels.append(c_name)

    X_clean_feats = np.array(X_clean_feats)
    corpus_labels = np.array(corpus_labels)

    print("\n3. Fitting PCA projection...")
    mu = np.mean(X_clean_feats, axis=0)
    sigma = np.std(X_clean_feats, axis=0) + 1e-8
    X_clean_std = (X_clean_feats - mu) / sigma
    U, S, Vt = np.linalg.svd(X_clean_std, full_matrices=False)
    Vt = Vt[:2]
    X_clean_pca = X_clean_std @ Vt.T

    print("\n4. Generating and evaluating perturbed frames...")
    perturbed_frames = []
    for i in range(N_total):
        c_name = corpus_labels[i]
        frame = all_corpora[c_name][i % N]
        p_idx = i % 12
        perturbed, _ = apply_random_perturbation(frame, sr, p_idx, rng)
        perturbed_frames.append(perturbed)

    X_all_coords = []
    Y_safety = []

    # Add clean frames (Safe/Works)
    for i in range(N_total):
        X_all_coords.append(X_clean_pca[i])
        Y_safety.append([1.0, 1.0, 1.0, 1.0, 1.0])

    # Evaluate perturbed frames
    for i in range(N_total):
        frame_pert = perturbed_frames[i]
        feats_pert = extract_physical_descriptors(frame_pert, sr)
        feats_std = (np.array(feats_pert) - mu) / sigma
        coords_proj = feats_std @ Vt.T
        X_all_coords.append(coords_proj)
        
        # Perturbed representations
        acf_p = compute_acf(frame_pert)
        cep_p = compute_cepstrum(frame_pert)
        mag_p = compute_stft(frame_pert, sr)
        if mag_p.ndim > 1: mag_p = np.mean(mag_p, axis=1)
        cqt_p = compute_cqt_frame(frame_pert, sr)
        cwt_p = compute_morlet_cwt(frame_pert, 64)
        
        # Clean reference representations
        acf_c, cep_c, mag_c, cqt_c, cwt_c = clean_representations[i]
        
        # Cosine similarities
        s_stft = cosine_similarity(mag_c, mag_p)
        s_acf  = cosine_similarity(acf_c, acf_p)
        s_cep  = cosine_similarity(cep_c, cep_p)
        s_cqt  = cosine_similarity(cqt_c, cqt_p)
        s_cwt  = cosine_similarity(cwt_c, cwt_p)
        
        # Pitch estimations
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

        p_cwt_c = max(1.0, estimate_pitch_cwt(cwt_c, sr))
        p_cwt_p = max(1.0, estimate_pitch_cwt(cwt_p, sr))
        e_cwt = np.clip(12.0 * np.abs(np.log2(p_cwt_p / p_cwt_c)), 0, 12.0)
        
        rep_labels = []
        for e, s in [(e_stft, s_stft), (e_acf, s_acf), (e_cep, s_cep), (e_cqt, s_cqt), (e_cwt, s_cwt)]:
            if e <= 1.0 and s >= 0.80:
                rep_labels.append(1.0)
            elif e <= 3.0 and s >= 0.60:
                rep_labels.append(0.5)
            else:
                rep_labels.append(0.0)
                
        Y_safety.append(rep_labels)

    X_all_coords = np.array(X_all_coords)
    Y_safety = np.array(Y_safety)

    print("\n5. Fitting Ridge Regression Safety Predictors...")
    X_poly = get_poly_features(X_all_coords)
    
    w_stft = train_ridge_regression(X_poly, Y_safety[:, 0], alpha=0.1)
    w_acf  = train_ridge_regression(X_poly, Y_safety[:, 1], alpha=0.1)
    w_cep  = train_ridge_regression(X_poly, Y_safety[:, 2], alpha=0.1)
    w_cqt  = train_ridge_regression(X_poly, Y_safety[:, 3], alpha=0.1)
    w_cwt  = train_ridge_regression(X_poly, Y_safety[:, 4], alpha=0.1)

    print("\n6. Exporting weights to src/framework/weights.py...")
    out_dir = os.path.join(project_root, "src", "framework")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "weights.py")

    with open(out_path, "w") as f:
        f.write("# Automatically generated by export_weights.py. Do not edit manually.\n")
        f.write("import numpy as np\n\n")
        
        f.write(f"MU = np.array({mu.tolist()})\n\n")
        f.write(f"SIGMA = np.array({sigma.tolist()})\n\n")
        f.write(f"VT = np.array({Vt.tolist()})\n\n")
        
        f.write(f"W_STFT = np.array({w_stft.tolist()})\n\n")
        f.write(f"W_ACF = np.array({w_acf.tolist()})\n\n")
        f.write(f"W_CEP = np.array({w_cep.tolist()})\n\n")
        f.write(f"W_CQT = np.array({w_cqt.tolist()})\n\n")
        f.write(f"W_CWT = np.array({w_cwt.tolist()})\n\n")

    print(f"Successfully exported all calibration weights to: {out_path}")
    print("=" * 75)

if __name__ == "__main__":
    run()
