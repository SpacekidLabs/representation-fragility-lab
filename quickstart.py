"""
Representation Intelligence Framework — Quickstart Guide
======================================================
This script demonstrates how a developer can drop the Representation Intelligence
Engine into their projects in just a few lines of code to analyze signal states
and adapt DSP parameters on-the-fly.

Installation:
-------------
$ pip install .
"""

import numpy as np
from representation_intelligence import RepresentationIntelligenceEngine

# 1. Initialize the engine (loads pre-trained state-space weights instantly)
print("Initializing Representation Intelligence Engine...")
engine = RepresentationIntelligenceEngine()

# 2. Simulate a frame of audio (e.g., 2048 samples at 16 kHz)
sr = 16000
duration_samples = 2048
t = np.arange(duration_samples) / sr

# Clean periodic vocal-like signal (low-entropy, harmonic structure)
clean_frame = np.sin(2 * np.pi * 220.0 * t) + 0.5 * np.sin(2 * np.pi * 440.0 * t)

# Noisy disordered signal (high-entropy, stochastically collapsed)
noisy_frame = clean_frame + np.random.normal(0, 1.5, duration_samples)

# 3. Analyze the clean frame
print("\n--- Analyzing Clean Harmonic Frame ---")
state_clean = engine.analyze(clean_frame, sr)
print(f"Coordinates (z1, z2): ({state_clean.coordinate[0]:.4f}, {state_clean.coordinate[1]:.4f})")
print(f"Detected Region:      '{state_clean.region}'")
print(f"Safe Representations: {state_clean.safe_representations}")
print(f"Recommended Window:   {state_clean.recommended_window} samples")

# 4. Analyze the noisy frame
print("\n--- Analyzing Noisy Frame ---")
state_noisy = engine.analyze(noisy_frame, sr)
print(f"Coordinates (z1, z2): ({state_noisy.coordinate[0]:.4f}, {state_noisy.coordinate[1]:.4f})")
print(f"Detected Region:      '{state_noisy.region}'")
print(f"Safe Representations: {state_noisy.safe_representations}")
print(f"Recommended Window:   {state_noisy.recommended_window} samples")

# 5. Adapt DSP parameters based on the engine state recommendation
print("\n--- Dynamic DSP Parameter Adaptation ---")
for name, state in [("Clean", state_clean), ("Noisy", state_noisy)]:
    # Access pre-calibrated parameters for common tasks
    denoising_params = state.recommended_parameters["denoising"]
    pitch_params = state.recommended_parameters["pitch_tracking"]
    
    print(f"[{name} Frame Parameters]:")
    print(f"  * Denoising alpha:      {denoising_params['alpha']:.4f}")
    print(f"  * Denoising beta:       {denoising_params['beta']:.4f}")
    print(f"  * YIN trough threshold: {pitch_params['yin_trough']:.4f}")
