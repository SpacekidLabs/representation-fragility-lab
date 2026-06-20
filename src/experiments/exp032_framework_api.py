"""
Experiment 032 — Framework API Verification
===========================================
Demonstrates and verifies the RepresentationIntelligenceEngine.
Feeds a continuous stream of vocal blocks perturbed with varying conditions,
tracks the engine's region classifications and recommendations, and plots
the resulting state space trajectory.
"""

import sys
import os
import warnings
import numpy as np
import scipy.signal
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import librosa

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

warnings.filterwarnings('ignore', category=UserWarning)

from src.framework.engine import RepresentationIntelligenceEngine
from src.experiments.exp028_failure_manifold_validation import extract_high_energy_frames

def run():
    print("=" * 75)
    print("EXPERIMENT 032 — FRAMEWORK API VERIFICATION")
    print("=" * 75)
    
    sr = 22050
    win = 1024
    
    print("\n1. Initializing RepresentationIntelligenceEngine...")
    engine = RepresentationIntelligenceEngine()
    print("Engine loaded successfully.")

    print("\n2. Loading vocal test audio and extracting frame sequence...")
    y_voc, _ = librosa.load(os.path.join(project_root, "Clean_vocal.wav"), sr=sr)
    # Extract 40 high-energy clean frames
    clean_frames = extract_high_energy_frames(y_voc, win, 40)
    
    # 3. Create a dynamic test sequence containing 4 distinct segments (10 frames each)
    # Segment 1: Clean (Healthy / Periodic Harmonic)
    # Segment 2: Noise (Noise Collapse)
    # Segment 3: Lowpass (Smooth Lowpass)
    # Segment 4: Hard clipping (Transient / Overloaded)
    print("\n3. Building dynamic test sequence (Clean -> Noisy -> Lowpass -> Clipped)...")
    test_frames = []
    conditions = []
    rng = np.random.default_rng(42)

    for idx, frame in enumerate(clean_frames):
        if idx < 10:
            # Clean
            perturbed = frame.copy()
            cond = "Clean (Periodic Harmonic)"
        elif idx < 20:
            # Noise
            perturbed = frame + rng.normal(0, 0.45, len(frame))
            cond = "Additive Noise (Noise Collapse)"
        elif idx < 30:
            # Lowpass
            b, a = scipy.signal.butter(4, 300.0 / (sr / 2.0), btype='low')
            perturbed = scipy.signal.filtfilt(b, a, frame)
            cond = "Lowpass Filtered (Smooth Lowpass)"
        else:
            # Clipped
            perturbed = np.clip(frame, -0.04, 0.04)
            cond = "Hard Clipped (Transient / Overloaded)"
            
        # Normalise energy
        perturbed /= np.max(np.abs(perturbed)) + 1e-9
        test_frames.append(perturbed)
        conditions.append(cond)

    # 4. Stream frames through the engine and log results
    print("\n4. Streaming frame blocks through framework.analyze()...")
    print(f"{'Frame':<5} | {'True Condition':<35} | {'Predicted Region':<22} | {'Coords':<16} | {'Primary Rep':<11} | {'Window Size':<11}")
    print("-" * 116)
    
    trajectory_coords = []
    state_regions = []
    
    for idx, frame in enumerate(test_frames):
        state = engine.analyze(frame, sr)
        trajectory_coords.append(state.coordinate)
        state_regions.append(state.region)
        
        # Log frame details
        coords_str = f"({state.coordinate[0]:.2f}, {state.coordinate[1]:.2f})"
        print(f"{idx:<5} | {conditions[idx]:<35} | {state.region:<22} | "
              f"{coords_str:<16} | "
              f"{state.recommendations['primary_representation']:<11} | "
              f"{state.recommendations['window_size']:<11}")
        
    trajectory_coords = np.array(trajectory_coords)

    # ---------------------------------------------------------------------------
    # Plotting
    # ---------------------------------------------------------------------------
    print("\n5. Plotting state space trajectory...")
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Scatter background points to show the coordinate space boundaries
    # We can plot the background clean frames from Exp 030/031
    # For visualization, we will plot the trajectory as a thick path with arrows
    colors = ["#2ca02c", "#d62728", "#ff7f0e", "#9467bd"]  # Green, Red, Orange, Purple
    segment_names = ["Clean", "Noisy", "Lowpass", "Clipped"]
    
    for seg_i in range(4):
        start = seg_i * 10
        end = (seg_i + 1) * 10
        ax.plot(trajectory_coords[start:end, 0], trajectory_coords[start:end, 1], 
                color=colors[seg_i], lw=3, label=f"Trajectory: {segment_names[seg_i]}")
        ax.scatter(trajectory_coords[start:end, 0], trajectory_coords[start:end, 1], 
                   color=colors[seg_i], s=80, edgecolors="white", zorder=5)
        # Add a text label at the start of each segment
        ax.text(trajectory_coords[start, 0] + 0.1, trajectory_coords[start, 1] + 0.1, 
                segment_names[seg_i], color=colors[seg_i], weight="bold", fontsize=10)
        
    # Draw arrows along the path
    for i in range(len(test_frames) - 1):
        if (i + 1) % 10 != 0: # Avoid drawing arrows across segment transitions
            ax.annotate("", xy=(trajectory_coords[i+1, 0], trajectory_coords[i+1, 1]),
                        xytext=(trajectory_coords[i, 0], trajectory_coords[i, 1]),
                        arrowprops=dict(arrowstyle="->", color="gray", lw=1.5, alpha=0.7))
            
    ax.set_title("Framework Trajectory in the Universal Audio State Space\n(Vocal frame sweeps through 4 physical states)", fontsize=13, fontweight="bold")
    ax.set_xlabel("PC1: Order ↔ Disorder")
    ax.set_ylabel("PC2: Harmonic ↔ Transient")
    ax.grid(True, alpha=0.08)
    ax.legend(loc="upper right")
    
    # Add region boundaries/labels on plot
    ax.text(-2.5, -1.0, "periodic_harmonic\n(Clean / Harmonic)", color="green", weight="bold", fontsize=9, bbox=dict(facecolor='black', alpha=0.6, edgecolor='none'))
    ax.text(2.0, 1.0, "noise_collapse\n(Uncorrelated noise)", color="red", weight="bold", fontsize=9, bbox=dict(facecolor='black', alpha=0.6, edgecolor='none'))
    ax.text(0.5, -1.2, "smooth_lowpass\n(Sine-like fundamental)", color="orange", weight="bold", fontsize=9, bbox=dict(facecolor='black', alpha=0.6, edgecolor='none'))
    ax.text(-2.0, 1.8, "transient_overloaded\n(Clipped / Clicks)", color="purple", weight="bold", fontsize=9, bbox=dict(facecolor='black', alpha=0.6, edgecolor='none'))

    out_dir = os.path.join(project_root, "results")
    out_path = os.path.join(out_dir, "exp032_framework_api.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    
    print(f"\nSaved Framework API verification plot: {out_path}")
    print("=" * 75)

if __name__ == "__main__":
    run()
