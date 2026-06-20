"""
Experiment 021 — Note Stabilization
=====================================
Identifies and fixes the root cause of the "sweeping" artifact in the
prototype auto-tuner.

Root causes confirmed by exp020 diagnostic:
  1. Target note re-computed independently every frame → target jumps
     between adjacent notes every 1–2 frames.
  2. Correction ratio therefore changes each frame (~±2%).
  3. Large Hanning windows (8192 samples) blended together during
     overlap-add turn those ratio changes into a continuous pitch glide.

Fix — NoteTracker:
  A simple state machine that enforces a minimum hold time before
  committing to a new target note.  Until a candidate note has been
  consistently detected for `min_hold_frames` consecutive frames, the
  previously committed note is used for correction instead.

Outputs:
  results/audio/real_clean_unstabilized.wav   — old behaviour (baseline)
  results/audio/real_clean_stabilized.wav     — with NoteTracker
  results/exp021_note_stabilization.png       — 3-panel diagnostic
"""

import sys
import os
import numpy as np
import scipy.signal
import scipy.io.wavfile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import librosa

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.representations.acf import compute_acf
from src.experiments.exp017_adaptive_routing import estimate_pitch_acf


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def save_wav(path, data, sr):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data_int16 = np.clip(data * 32767.0, -32768.0, 32767.0).astype(np.int16)
    scipy.io.wavfile.write(path, sr, data_int16)


def pitch_to_midi(p):
    """Return nearest MIDI note number for frequency p (Hz)."""
    return int(np.round(12.0 * np.log2(p / 440.0) + 69))


def midi_to_freq(midi):
    """Return frequency (Hz) for a MIDI note number."""
    return 440.0 * (2.0 ** ((midi - 69) / 12.0))


# ---------------------------------------------------------------------------
# NoteTracker — the core fix
# ---------------------------------------------------------------------------

class NoteTracker:
    """
    Enforces a minimum hold time before switching to a new target note.

    Parameters
    ----------
    min_hold_frames : int
        Number of consecutive frames a candidate note must be detected
        before it replaces the current committed note.
    """

    def __init__(self, min_hold_frames: int = 3):
        self.min_hold_frames = min_hold_frames
        self._committed_midi: int | None = None   # currently locked note
        self._candidate_midi: int | None = None   # note being evaluated
        self._candidate_count: int = 0            # consecutive frame count

    def update(self, detected_pitch: float) -> float:
        """
        Feed one frame's detected pitch.  Returns the stabilized target
        frequency (Hz) to use for correction this frame.
        """
        if not (80.0 <= detected_pitch <= 1200.0):
            # Unvoiced / out-of-range — hold current commitment
            return midi_to_freq(self._committed_midi) if self._committed_midi else detected_pitch

        new_midi = pitch_to_midi(detected_pitch)

        if self._committed_midi is None:
            # Bootstrap on first voiced frame
            self._committed_midi = new_midi
            self._candidate_midi = new_midi
            self._candidate_count = 1
            return midi_to_freq(new_midi)

        if new_midi == self._committed_midi:
            # Detected note agrees with current commitment — reset candidate
            self._candidate_midi = new_midi
            self._candidate_count = 1
        else:
            # Detected note differs from commitment
            if new_midi == self._candidate_midi:
                self._candidate_count += 1
            else:
                # New challenger — restart candidate counter
                self._candidate_midi = new_midi
                self._candidate_count = 1

            # Promote candidate once it has held long enough
            if self._candidate_count >= self.min_hold_frames:
                self._committed_midi = self._candidate_midi
                self._candidate_count = 1

        return midi_to_freq(self._committed_midi)


# ---------------------------------------------------------------------------
# Core: run one pass of the tuner
# ---------------------------------------------------------------------------

def run_tuner(audio: np.ndarray, sr: int, stabilize: bool,
              min_hold_frames: int = 3,
              win_size: int = 4096, hop_size: int = 512) -> tuple:
    """
    Run block-by-block pitch correction.

    Returns
    -------
    output_audio : np.ndarray
    log_detected : list[float]   — raw detected pitch per frame
    log_target   : list[float]   — stabilized (or raw) target per frame
    log_ratio    : list[float]   — correction ratio per frame
    log_shift_st : list[float]   — pitch shift in semitones per frame
    t_frames     : np.ndarray    — centre timestamp of each frame (s)
    """
    tracker = NoteTracker(min_hold_frames=min_hold_frames) if stabilize else None

    output_audio = np.zeros(len(audio) + win_size)
    window_sum   = np.zeros_like(output_audio)

    min_lag = int(sr / 1000)
    max_lag = int(sr / 80)

    log_detected, log_target, log_ratio, log_shift_st = [], [], [], []

    num_blocks = (len(audio) - win_size) // hop_size + 1

    for n in range(num_blocks):
        start = n * hop_size
        end   = start + win_size
        block = audio[start:end]

        # Pitch detection on the full block (smaller window = sharper estimate)
        frame = block * np.hanning(win_size)
        acf   = compute_acf(frame)
        p, _  = estimate_pitch_acf(acf, sr)

        # Target note selection
        if stabilize:
            f_target = tracker.update(p)
        else:
            if 80.0 <= p <= 1200.0:
                f_target = midi_to_freq(pitch_to_midi(p))
            else:
                f_target = p

        # Semitone shift
        if 80.0 <= p <= 1200.0 and 80.0 <= f_target <= 1200.0:
            s = np.clip(12.0 * np.log2(f_target / p), -6.0, 6.0)
        else:
            s = 0.0

        ratio = f_target / p if p > 0 else 1.0

        log_detected.append(p)
        log_target.append(f_target)
        log_ratio.append(ratio)
        log_shift_st.append(s)

        # Pitch shift + overlap-add
        shifted = librosa.effects.pitch_shift(block, sr=sr, n_steps=s)
        win     = np.hanning(win_size)
        output_audio[start:end] += shifted * win
        window_sum[start:end]   += win

    # Normalise overlap-add
    mask = window_sum > 1e-5
    output_audio[mask] /= window_sum[mask]
    output_audio = output_audio[:len(audio)]
    if np.max(np.abs(output_audio)) > 0:
        output_audio /= np.max(np.abs(output_audio))
        output_audio *= 0.95

    t_frames = np.array([(n * hop_size + win_size // 2) / sr
                         for n in range(num_blocks)])

    return output_audio, log_detected, log_target, log_ratio, log_shift_st, t_frames


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_note_stabilization_experiment():
    print("=" * 60)
    print("EXPERIMENT 021 — NOTE STABILIZATION")
    print("=" * 60)

    sr = 22050
    vocal_path = os.path.join(project_root, "Clean_vocal.wav")
    print(f"Loading: {vocal_path}")
    audio, _ = librosa.load(vocal_path, sr=sr)
    audio /= np.max(np.abs(audio))
    print(f"Duration: {len(audio)/sr:.2f}s")

    # Smaller window = sharper per-frame estimate, less blending artifact
    WIN  = 4096   # ~185 ms  (halved from exp020's 8192)
    HOP  = 256    # ~11.6 ms (quartered — tighter time resolution)
    HOLD = 3      # frames — ~35 ms minimum note hold before switching

    print(f"\nWindow: {WIN} samples ({WIN/sr*1000:.0f} ms)")
    print(f"Hop:    {HOP} samples ({HOP/sr*1000:.1f} ms)")
    print(f"Hold:   {HOLD} frames ({HOLD*HOP/sr*1000:.0f} ms minimum note duration)\n")

    # Run both variants
    print("Running UNSTABILIZED tuner (baseline — old behaviour)...")
    out_unstab, det_u, tgt_u, rat_u, sst_u, t_u = run_tuner(
        audio, sr, stabilize=False, win_size=WIN, hop_size=HOP)

    print("Running STABILIZED tuner (NoteTracker fix)...")
    out_stab, det_s, tgt_s, rat_s, sst_s, t_s = run_tuner(
        audio, sr, stabilize=True, min_hold_frames=HOLD, win_size=WIN, hop_size=HOP)

    # Save audio
    save_wav(os.path.join(project_root, "results", "audio", "real_clean_unstabilized.wav"), out_unstab, sr)
    save_wav(os.path.join(project_root, "results", "audio", "real_clean_stabilized.wav"),   out_stab,   sr)
    print("\nSaved audio outputs.")

    # ------------------------------------------------------------------
    # Quantify target note jitter
    # ------------------------------------------------------------------
    def count_target_changes(targets):
        arr = np.array(targets)
        return int(np.sum(np.diff(arr.round(1)) != 0))

    jitter_u = count_target_changes(tgt_u)
    jitter_s = count_target_changes(tgt_s)
    print(f"\nTarget note changes (unstabilized): {jitter_u} over {len(tgt_u)} frames")
    print(f"Target note changes (stabilized)  : {jitter_s} over {len(tgt_s)} frames")
    print(f"Jitter reduction: {(1 - jitter_s/jitter_u)*100:.1f}%")

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    print("\nGenerating diagnostic plot...")
    plt.style.use("dark_background")
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    plt.subplots_adjust(hspace=0.08)

    note_names = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]

    # Panel 1 — Detected vs target (both variants)
    ax1 = axes[0]
    ax1.plot(t_u, det_u, color="#888888", lw=1.2, alpha=0.7, label="Detected Pitch")
    ax1.step(t_u, tgt_u, color="#ff7f00", lw=1.5, where="mid", alpha=0.8, label="Target — Unstabilized")
    ax1.step(t_s, tgt_s, color="#33a02c", lw=2.2, where="mid",             label="Target — Stabilized")
    ax1.set_ylabel("Frequency (Hz)")
    ax1.set_title("Detected Pitch vs Target Note: Unstabilized vs Stabilized", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=9, loc="upper right")
    ax1.grid(True, alpha=0.15)

    # Add note name annotations on right axis
    ax1r = ax1.twinx()
    unique_midi = sorted(set(pitch_to_midi(f) for f in tgt_s if 80 < f < 1200))
    for m in unique_midi:
        f = midi_to_freq(m)
        name = note_names[m % 12] + str(m // 12 - 1)
        ax1r.axhline(f, color="white", alpha=0.08, lw=0.6, linestyle="--")
        ax1r.text(t_s[-1] + 0.01, f, name, va="center", fontsize=7, color="white", alpha=0.5)
    ax1r.set_ylim(ax1.get_ylim())
    ax1r.set_yticks([])

    # Panel 2 — Correction ratio per frame (the glide generator)
    ax2 = axes[1]
    ax2.step(t_u, rat_u, color="#ff7f00", lw=1.2, where="mid", alpha=0.7, label="Ratio — Unstabilized")
    ax2.step(t_s, rat_s, color="#33a02c", lw=1.8, where="mid",             label="Ratio — Stabilized")
    ax2.axhline(1.0, color="white", alpha=0.25, lw=0.8)
    ax2.set_ylabel("Correction Ratio\n(target / detected)")
    ax2.set_title("Pitch Correction Ratio per Frame — Oscillation = Sweep Cause", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.15)

    # Panel 3 — Shift in semitones
    ax3 = axes[2]
    ax3.step(t_u, sst_u, color="#ff7f00", lw=1.2, where="mid", alpha=0.7, label="Shift (st) — Unstabilized")
    ax3.step(t_s, sst_s, color="#33a02c", lw=1.8, where="mid",             label="Shift (st) — Stabilized")
    ax3.axhline(0, color="white", alpha=0.25, lw=0.8)
    ax3.set_ylabel("Pitch Shift (semitones)")
    ax3.set_xlabel("Time (seconds)")
    ax3.set_title("Correction in Semitones per Frame", fontsize=12, fontweight="bold")
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.15)

    # Annotate jitter reduction
    fig.text(0.99, 0.01,
             f"Target note changes — Unstabilized: {jitter_u}   Stabilized: {jitter_s}   "
             f"({(1-jitter_s/jitter_u)*100:.0f}% jitter reduction)",
             ha="right", va="bottom", fontsize=9, color="#aaaaaa")

    out_plot = os.path.join(project_root, "results", "exp021_note_stabilization.png")
    plt.savefig(out_plot, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Plot saved: {out_plot}")
    print("=" * 60)


if __name__ == "__main__":
    run_note_stabilization_experiment()
