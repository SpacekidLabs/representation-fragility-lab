"""
Experiment 022 — Confidence-Gated Note Tracking
================================================
The NoteTracker from Exp 021 locked onto bad notes during low-confidence
regions (attack transients, breath noise).  The fix is to gate note
commitment on the ACF self-confidence signal from Exp 015.

Three-State NoteTracker
-----------------------
  UNDECIDED
    confidence < LOCK_THRESH
    → don't commit, don't correct
    → just observe

  TRACKING
    confidence >= LOCK_THRESH but not yet held for LOCK_FRAMES
    → collecting evidence, building frame count
    → still no correction (don't correct on partial evidence)

  LOCKED
    confidence >= LOCK_THRESH for LOCK_FRAMES consecutive frames
    → commit to note, apply full correction

Unlock rule:
  LOCKED → UNDECIDED  if confidence < UNLOCK_THRESH for UNLOCK_FRAMES

Comparison:
  Version A  — Unstabilized (original exp020 bug)
  Version B  — Exp 021 NoteTracker (no confidence gating)
  Version C  — Confidence-Gated NoteTracker (this experiment)

Outputs:
  results/audio/real_clean_conf_gated.wav
  results/exp022_confidence_gated.png
"""

import sys
import os
import enum
import numpy as np
import scipy.signal
import scipy.io.wavfile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import librosa

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.representations.acf import compute_acf
from src.representations.cepstrum import compute_cepstrum
from src.experiments.exp017_adaptive_routing import estimate_pitch_acf


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def save_wav(path, data, sr):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data_int16 = np.clip(data * 32767.0, -32768.0, 32767.0).astype(np.int16)
    scipy.io.wavfile.write(path, sr, data_int16)


def pitch_to_midi(p):
    return int(np.round(12.0 * np.log2(p / 440.0) + 69))


def midi_to_freq(midi):
    return 440.0 * (2.0 ** ((midi - 69) / 12.0))


def acf_confidence(acf: np.ndarray, min_lag: int, max_lag: int) -> float:
    """
    Peak prominence ratio — the self-confidence signal from Exp 015.
    High for clean periodic signals.  Near-zero for noise / transients.
    """
    if acf[0] < 1e-9:
        return 0.0
    peak = np.max(acf[min_lag:max_lag])
    ratio = peak / acf[0]
    return float(np.clip((ratio - 0.15) / (0.75 - 0.15), 0.0, 1.0))


# ---------------------------------------------------------------------------
# Three-State NoteTracker
# ---------------------------------------------------------------------------

class State(enum.Enum):
    UNDECIDED = 0
    TRACKING  = 1
    LOCKED    = 2


class ConfidenceGatedNoteTracker:
    """
    Parameters
    ----------
    lock_thresh    : float  confidence required to enter / hold TRACKING
    lock_frames    : int    consecutive high-conf frames before LOCKED
    unlock_thresh  : float  confidence below which we count toward unlock
    unlock_frames  : int    consecutive low-conf frames before reverting
    """

    def __init__(self,
                 lock_thresh:   float = 0.60,
                 lock_frames:   int   = 3,
                 unlock_thresh: float = 0.35,
                 unlock_frames: int   = 3):
        self.lock_thresh   = lock_thresh
        self.lock_frames   = lock_frames
        self.unlock_thresh = unlock_thresh
        self.unlock_frames = unlock_frames

        self.state: State = State.UNDECIDED

        self._locked_midi:      int | None = None  # committed note
        self._candidate_midi:   int | None = None  # note being evaluated
        self._track_count:      int = 0            # consecutive high-conf frames
        self._unlock_count:     int = 0            # consecutive low-conf frames

    def update(self, detected_pitch: float, confidence: float) -> tuple[float | None, State]:
        """
        Returns
        -------
        target_freq : float | None
            Frequency to correct toward, or None if no correction this frame.
        state : State
            Current state after update.
        """
        voiced = 80.0 <= detected_pitch <= 1200.0
        high_conf = voiced and confidence >= self.lock_thresh
        low_conf  = (not voiced) or (confidence < self.unlock_thresh)

        # ------------------------------------------------------------------
        if self.state == State.UNDECIDED:
            self._unlock_count = 0
            if high_conf:
                # Start tracking a candidate
                new_midi = pitch_to_midi(detected_pitch)
                if new_midi == self._candidate_midi:
                    self._track_count += 1
                else:
                    self._candidate_midi = new_midi
                    self._track_count = 1

                if self._track_count >= self.lock_frames:
                    # Promote to LOCKED
                    self._locked_midi = self._candidate_midi
                    self.state = State.LOCKED
                    return midi_to_freq(self._locked_midi), self.state
                else:
                    self.state = State.TRACKING
                    return None, self.state
            else:
                self._track_count = 0
                self._candidate_midi = None
                return None, self.state   # undecided → no correction

        # ------------------------------------------------------------------
        elif self.state == State.TRACKING:
            self._unlock_count = 0
            if high_conf:
                new_midi = pitch_to_midi(detected_pitch)
                if new_midi == self._candidate_midi:
                    self._track_count += 1
                else:
                    self._candidate_midi = new_midi
                    self._track_count = 1

                if self._track_count >= self.lock_frames:
                    self._locked_midi = self._candidate_midi
                    self.state = State.LOCKED
                    return midi_to_freq(self._locked_midi), self.state
                else:
                    return None, self.state  # still building evidence
            else:
                # Confidence dropped before we locked — back to UNDECIDED
                self.state = State.UNDECIDED
                self._track_count = 0
                self._candidate_midi = None
                return None, self.state

        # ------------------------------------------------------------------
        elif self.state == State.LOCKED:
            if low_conf:
                self._unlock_count += 1
                if self._unlock_count >= self.unlock_frames:
                    self.state = State.UNDECIDED
                    self._track_count = 0
                    self._candidate_midi = None
                    self._unlock_count = 0
                    return None, self.state
            else:
                self._unlock_count = 0
                # Check if pitch has moved to a new stable note
                if high_conf:
                    new_midi = pitch_to_midi(detected_pitch)
                    if new_midi != self._locked_midi:
                        if new_midi == self._candidate_midi:
                            self._track_count += 1
                        else:
                            self._candidate_midi = new_midi
                            self._track_count = 1
                        if self._track_count >= self.lock_frames:
                            self._locked_midi = self._candidate_midi
                            self._track_count = 1

            # Still locked — correct toward committed note
            return midi_to_freq(self._locked_midi), self.state

        return None, self.state


# ---------------------------------------------------------------------------
# Simple NoteTracker from Exp 021 (for comparison)
# ---------------------------------------------------------------------------

class SimpleNoteTracker:
    def __init__(self, min_hold_frames: int = 3):
        self.min_hold_frames = min_hold_frames
        self._committed_midi: int | None = None
        self._candidate_midi: int | None = None
        self._candidate_count: int = 0

    def update(self, detected_pitch: float) -> float:
        if not (80.0 <= detected_pitch <= 1200.0):
            return midi_to_freq(self._committed_midi) if self._committed_midi else detected_pitch
        new_midi = pitch_to_midi(detected_pitch)
        if self._committed_midi is None:
            self._committed_midi = new_midi
            self._candidate_midi = new_midi
            self._candidate_count = 1
            return midi_to_freq(new_midi)
        if new_midi == self._committed_midi:
            self._candidate_midi = new_midi
            self._candidate_count = 1
        else:
            if new_midi == self._candidate_midi:
                self._candidate_count += 1
            else:
                self._candidate_midi = new_midi
                self._candidate_count = 1
            if self._candidate_count >= self.min_hold_frames:
                self._committed_midi = self._candidate_midi
                self._candidate_count = 1
        return midi_to_freq(self._committed_midi)


# ---------------------------------------------------------------------------
# Tuner runner
# ---------------------------------------------------------------------------

def run_tuner(audio: np.ndarray, sr: int, mode: str,
              win_size: int = 4096, hop_size: int = 256) -> dict:
    """
    mode: 'unstabilized' | 'simple' | 'confidence_gated'
    """
    min_lag = int(sr / 1000)
    max_lag = int(sr / 80)

    tracker_simple = SimpleNoteTracker(min_hold_frames=3) if mode == "simple" else None
    tracker_conf   = ConfidenceGatedNoteTracker() if mode == "confidence_gated" else None

    output_audio = np.zeros(len(audio) + win_size)
    window_sum   = np.zeros_like(output_audio)

    log = dict(t=[], detected=[], confidence=[], target=[], shift_st=[], state=[])

    num_blocks = (len(audio) - win_size) // hop_size + 1

    for n in range(num_blocks):
        start = n * hop_size
        block = audio[start:start + win_size]
        frame = block * np.hanning(win_size)

        acf  = compute_acf(frame)
        conf = acf_confidence(acf, min_lag, max_lag)
        p, _ = estimate_pitch_acf(acf, sr)

        # ------------------------------------------------------------------
        state_label = State.UNDECIDED

        if mode == "unstabilized":
            if 80.0 <= p <= 1200.0:
                f_target = midi_to_freq(pitch_to_midi(p))
                state_label = State.LOCKED
            else:
                f_target = None

        elif mode == "simple":
            f_target = tracker_simple.update(p)
            state_label = State.LOCKED

        elif mode == "confidence_gated":
            f_target, state_label = tracker_conf.update(p, conf)

        # ------------------------------------------------------------------
        # Pitch shift
        if f_target is not None and 80.0 <= p <= 1200.0:
            s = np.clip(12.0 * np.log2(f_target / p), -6.0, 6.0)
        else:
            s = 0.0
            f_target = p  # log the detected pitch as target for plotting

        log["t"].append((n * hop_size + win_size // 2) / sr)
        log["detected"].append(p)
        log["confidence"].append(conf)
        log["target"].append(f_target)
        log["shift_st"].append(s)
        log["state"].append(state_label)

        shifted = librosa.effects.pitch_shift(block, sr=sr, n_steps=s)
        win = np.hanning(win_size)
        output_audio[start:start + win_size] += shifted * win
        window_sum[start:start + win_size]   += win

    mask = window_sum > 1e-5
    output_audio[mask] /= window_sum[mask]
    output_audio = output_audio[:len(audio)]
    if np.max(np.abs(output_audio)) > 0:
        output_audio /= np.max(np.abs(output_audio))
        output_audio *= 0.95

    return {"audio": output_audio, **{k: np.array(v) for k, v in log.items()}}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run():
    print("=" * 60)
    print("EXPERIMENT 022 — CONFIDENCE-GATED NOTE TRACKING")
    print("=" * 60)

    sr = 22050
    vocal_path = os.path.join(project_root, "Clean_vocal.wav")
    audio, _ = librosa.load(vocal_path, sr=sr)
    audio /= np.max(np.abs(audio))
    print(f"Loaded: {len(audio)/sr:.2f}s  ({sr} Hz)")

    WIN, HOP = 4096, 256

    print("\nRunning UNSTABILIZED (original)...")
    res_u = run_tuner(audio, sr, "unstabilized",   win_size=WIN, hop_size=HOP)

    print("Running SIMPLE NoteTracker (Exp 021)...")
    res_s = run_tuner(audio, sr, "simple",          win_size=WIN, hop_size=HOP)

    print("Running CONFIDENCE-GATED NoteTracker (Exp 022)...")
    res_c = run_tuner(audio, sr, "confidence_gated", win_size=WIN, hop_size=HOP)

    # Save audio
    audio_dir = os.path.join(project_root, "results", "audio")
    save_wav(os.path.join(audio_dir, "real_clean_conf_gated.wav"), res_c["audio"], sr)
    print("\nSaved: real_clean_conf_gated.wav")

    # ------------------------------------------------------------------
    # Count correction-applied frames
    def active_frames(log):
        return int(np.sum(np.abs(log["shift_st"]) > 0.01))

    def target_changes(log):
        arr = log["target"].round(1)
        return int(np.sum(np.diff(arr) != 0))

    print(f"\n{'':30s}  {'Unstab':>8}  {'Simple':>8}  {'Conf-Gated':>10}")
    print(f"{'Target note changes':30s}  {target_changes(res_u):>8}  {target_changes(res_s):>8}  {target_changes(res_c):>10}")
    print(f"{'Active correction frames':30s}  {active_frames(res_u):>8}  {active_frames(res_s):>8}  {active_frames(res_c):>10}")
    print(f"{'Total frames':30s}  {len(res_u['t']):>8}  {len(res_s['t']):>8}  {len(res_c['t']):>10}")

    # State breakdown for confidence-gated
    states = res_c["state"]
    n_total = len(states)
    n_undecided = np.sum(states == State.UNDECIDED)
    n_tracking  = np.sum(states == State.TRACKING)
    n_locked    = np.sum(states == State.LOCKED)
    print(f"\nConf-Gated state breakdown:")
    print(f"  UNDECIDED : {n_undecided:4d} frames ({100*n_undecided/n_total:.1f}%)")
    print(f"  TRACKING  : {n_tracking:4d} frames ({100*n_tracking/n_total:.1f}%)")
    print(f"  LOCKED    : {n_locked:4d} frames ({100*n_locked/n_total:.1f}%)")

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    print("\nGenerating plot...")

    STATE_COLORS = {
        State.UNDECIDED: "#ff4444",   # red
        State.TRACKING:  "#ffaa00",   # amber
        State.LOCKED:    "#33cc66",   # green
    }

    plt.style.use("dark_background")
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    plt.subplots_adjust(hspace=0.08)

    t_u, t_s, t_c = res_u["t"], res_s["t"], res_c["t"]

    # Panel 1 — ACF self-confidence over time
    ax0 = axes[0]
    ax0.fill_between(t_c, res_c["confidence"], alpha=0.35, color="#377eb8")
    ax0.plot(t_c, res_c["confidence"], color="#377eb8", lw=1.5, label="ACF Confidence")
    ax0.axhline(0.60, color="#33cc66", lw=1.0, linestyle="--", alpha=0.7, label="Lock threshold (0.60)")
    ax0.axhline(0.35, color="#ff4444", lw=1.0, linestyle="--", alpha=0.7, label="Unlock threshold (0.35)")
    ax0.set_ylabel("Confidence")
    ax0.set_ylim(-0.05, 1.15)
    ax0.set_title("ACF Self-Confidence and NoteTracker State Gating", fontsize=12, fontweight="bold")
    ax0.legend(fontsize=8, loc="upper right")
    ax0.grid(True, alpha=0.12)

    # Shade background by state
    for i in range(len(t_c) - 1):
        ax0.axvspan(t_c[i], t_c[i+1], color=STATE_COLORS[res_c["state"][i]], alpha=0.08)

    # Panel 2 — Target note: all three variants
    ax1 = axes[1]
    ax1.plot(t_u, res_u["detected"], color="#555555", lw=1.0, alpha=0.6, label="Detected Pitch")
    ax1.step(t_u, res_u["target"], color="#ff7f00", lw=1.4, where="mid", alpha=0.7, label="Target — Unstabilized")
    ax1.step(t_s, res_s["target"], color="#e7298a", lw=1.4, where="mid", alpha=0.7, label="Target — Simple Tracker")
    ax1.step(t_c, res_c["target"], color="#33cc66", lw=2.2, where="mid",             label="Target — Confidence-Gated")

    # Shade UNDECIDED / TRACKING / LOCKED regions in panel 2
    for i in range(len(t_c) - 1):
        ax1.axvspan(t_c[i], t_c[i+1], color=STATE_COLORS[res_c["state"][i]], alpha=0.06)

    ax1.set_ylabel("Frequency (Hz)")
    ax1.set_title("Target Note Decisions: Three Tuner Variants", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=8, loc="upper right")
    ax1.grid(True, alpha=0.12)

    # Panel 3 — Pitch shift in semitones
    ax2 = axes[2]
    ax2.step(t_u, res_u["shift_st"], color="#ff7f00", lw=1.2, where="mid", alpha=0.6, label="Shift — Unstabilized")
    ax2.step(t_s, res_s["shift_st"], color="#e7298a", lw=1.2, where="mid", alpha=0.6, label="Shift — Simple Tracker")
    ax2.step(t_c, res_c["shift_st"], color="#33cc66", lw=2.0, where="mid",             label="Shift — Confidence-Gated")
    ax2.axhline(0, color="white", alpha=0.15, lw=0.8)
    for i in range(len(t_c) - 1):
        ax2.axvspan(t_c[i], t_c[i+1], color=STATE_COLORS[res_c["state"][i]], alpha=0.06)
    ax2.set_ylabel("Pitch Shift (semitones)")
    ax2.set_title("Applied Correction per Frame", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=8, loc="upper right")
    ax2.grid(True, alpha=0.12)

    # Panel 4 — State timeline
    ax3 = axes[3]
    state_vals = np.array([s.value for s in res_c["state"]], dtype=float)
    for i in range(len(t_c) - 1):
        ax3.axvspan(t_c[i], t_c[i+1],
                    color=STATE_COLORS[res_c["state"][i]], alpha=0.55)
    ax3.step(t_c, state_vals, color="white", lw=1.0, where="mid", alpha=0.5)
    ax3.set_yticks([0, 1, 2])
    ax3.set_yticklabels(["UNDECIDED", "TRACKING", "LOCKED"], fontsize=8)
    ax3.set_ylabel("NoteTracker State")
    ax3.set_xlabel("Time (seconds)")
    ax3.set_title("Confidence-Gated NoteTracker State Machine Timeline", fontsize=12, fontweight="bold")
    ax3.grid(True, alpha=0.08)
    ax3.set_ylim(-0.3, 2.4)

    # Legend patches for states
    patches = [mpatches.Patch(color=c, label=s.name, alpha=0.7)
               for s, c in STATE_COLORS.items()]
    ax3.legend(handles=patches, fontsize=8, loc="upper right")

    # Footer stats
    fig.text(0.99, 0.005,
             f"UNDECIDED: {n_undecided} frames ({100*n_undecided/n_total:.0f}%)  "
             f"TRACKING: {n_tracking} frames ({100*n_tracking/n_total:.0f}%)  "
             f"LOCKED: {n_locked} frames ({100*n_locked/n_total:.0f}%)",
             ha="right", va="bottom", fontsize=8, color="#888888")

    out_path = os.path.join(project_root, "results", "exp022_confidence_gated.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Plot saved: {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    run()
