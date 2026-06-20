import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add the project root to the python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.signals import sine, harmonic_stack, fm_tone, chirp, impulse, noise
from src.representations.stft import compute_stft
from src.representations.acf import compute_acf
from src.representations.cepstrum import compute_cepstrum

def run_baseline_experiment():
    sr = 22050
    duration = 1.0
    
    # 1. Generate all signals
    classes = [
        {"name": "Sine", "signal": sine.generate(duration, sr)[0]},
        {"name": "Harmonic Stack", "signal": harmonic_stack.generate(duration, sr)[0]},
        {"name": "FM Tone", "signal": fm_tone.generate(duration, sr)[0]},
        {"name": "Chirp", "signal": chirp.generate(duration, sr)[0]},
        {"name": "Impulse", "signal": impulse.generate(duration, sr, delay_s=0.0)[0]},
        {"name": "White Noise", "signal": noise.generate(duration, sr)[0]},
    ]
    
    # Grid setup: 6 rows (signal classes), 4 columns (Waveform, STFT, ACF, Cepstrum)
    fig, axes = plt.subplots(6, 4, figsize=(16, 18), sharex=False, sharey=False)
    plt.subplots_adjust(hspace=0.5, wspace=0.3)
    
    print("Computing baseline representations for all signal classes...")
    
    for row_idx, cls in enumerate(classes):
        name = cls["name"]
        sig = cls["signal"]
        
        # Compute representations
        stft_mag = compute_stft(sig, sr)
        acf = compute_acf(sig)
        cepstrum = compute_cepstrum(sig)
        
        # --- Column 1: Waveform (First 1000 samples) ---
        ax_wave = axes[row_idx, 0]
        ax_wave.plot(sig[:1000], color='#2ca02c')
        ax_wave.set_title(f"{name} (Waveform)", fontsize=10, fontweight='bold')
        ax_wave.grid(True, linestyle='--', alpha=0.5)
        ax_wave.set_ylim(-1.1, 1.1)
        
        # --- Column 2: STFT (Time-Frequency Spectrogram) ---
        ax_stft = axes[row_idx, 1]
        limit_bins = min(150, stft_mag.shape[0])
        ax_stft.imshow(stft_mag[:limit_bins, :], aspect='auto', origin='lower', cmap='magma')
        ax_stft.set_title(f"{name} (STFT)", fontsize=10, fontweight='bold')
        
        # --- Column 3: ACF (First 1000 lags) ---
        ax_acf = axes[row_idx, 2]
        ax_acf.plot(acf[:1000], color='#ff7f0e')
        ax_acf.set_title(f"{name} (ACF)", fontsize=10, fontweight='bold')
        ax_acf.grid(True, linestyle='--', alpha=0.5)
        
        # --- Column 4: Cepstrum (First 500 coefficients) ---
        ax_cep = axes[row_idx, 3]
        ax_cep.plot(cepstrum[:500], color='#1f77b4')
        ax_cep.set_title(f"{name} (Cepstrum)", fontsize=10, fontweight='bold')
        ax_cep.grid(True, linestyle='--', alpha=0.5)
        
    os.makedirs(os.path.join(project_root, "results"), exist_ok=True)
    output_path = os.path.join(project_root, "results", "exp009_signal_class_baselines.png")
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"Baseline comparison plot saved to: {output_path}")

if __name__ == "__main__":
    run_baseline_experiment()
