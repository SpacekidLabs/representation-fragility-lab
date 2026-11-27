"""
A learned pitch lens: CREPE (issue #26, Exp 052)
================================================
``CrepePitch`` wraps the pretrained CREPE network (Kim et al. 2018, the
``torchcrepe`` port) as a core ``Representation``, so it is compared, voted and
patterned exactly like the hand-built lenses in ``framework/audio.py``.

Pinned for reproducibility: torch 2.15.0, torchcrepe 0.0.24, model "full"
(the weights ship inside the torchcrepe package; ``weights_sha256`` reports the
hash of the file actually loaded).

How a frame is read
  1. resample the frame to 16 kHz (CREPE's rate) with ``scipy.signal.resample_poly``
  2. take the centre 1024 samples (64 ms) and normalise them (zero mean, unit std),
     as CREPE does per window
  3. the network gives 360 pitch-bin probabilities (20 cents apart, 32 Hz–2 kHz)
  4. bins outside [fmin, fmax] are dropped (80–1000 Hz, the classic lenses' range)
  5. f0 = the original CREPE decoding: weighted mean of the bins within ±4 of the
     peak, in cents; confidence = the peak probability ("periodicity"), mapped to
     0..1 with the calibration rule of the CQT/wavelet lenses (see CALIBRATION).

torch is imported lazily: importing this module, or the audio medium, never
needs it.
"""

from __future__ import annotations

import hashlib
import os
from typing import Any, Mapping

import numpy as np
from scipy.signal import resample_poly

from .audio import Reading, Representation
from .disagreement import DEFAULT_FMAX, DEFAULT_FMIN

CREPE_SR = 16000
CREPE_WINDOW = 1024
MODEL = "full"
PINNED = {"torch": "2.15.0", "torchcrepe": "0.0.24", "model": MODEL}

# Bin i is at 20 i + 1997.379... cents above 10 Hz (CREPE's mapping).
_CENTS = 20.0 * np.arange(360) + 1997.3794084376191
_HZ = 10.0 * 2.0 ** (_CENTS / 1200.0)

# Peak probability → confidence: lo = highest peak over 60 white-noise frames,
# hi = 5th percentile over clean synthetic frames (Exp 042 signals). Same rule as
# the CQT / wavelet lenses. Set by src/experiments/exp052_learned_lens.py --calibrate.
CALIBRATION = (0.105, 0.872)


def _scale(raw: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip((np.asarray(raw, float) - lo) / (hi - lo), 0.0, 1.0)


class _Crepe:
    """Lazy holder of the torch model (loaded once per process)."""

    model = None
    device = "cpu"

    @classmethod
    def get(cls):
        if cls.model is None:
            import torch
            import torchcrepe

            torch.set_grad_enabled(False)
            m = torchcrepe.Crepe(MODEL)
            path = cls.weights_path()
            m.load_state_dict(torch.load(path, map_location="cpu"))
            m.eval()
            # CPU by default. RFL_CREPE_DEVICE=mps uses the Apple GPU (same outputs to
            # ~2e-6, ~2.5x faster), but it hung in a long run of Exp 052.
            cls.device = os.environ.get("RFL_CREPE_DEVICE", "cpu")
            cls.model = m.to(cls.device)
        return cls.model

    @staticmethod
    def weights_path() -> str:
        import torchcrepe

        return os.path.join(os.path.dirname(torchcrepe.__file__), "assets", f"{MODEL}.pth")

    @classmethod
    def weights_sha256(cls) -> str:
        with open(cls.weights_path(), "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()


def crepe_windows(frames: np.ndarray, sr: int) -> np.ndarray:
    """(n, frame_len) frames at ``sr`` → (n, 1024) normalised CREPE windows."""
    frames = np.atleast_2d(np.asarray(frames, float))
    g = np.gcd(int(sr), CREPE_SR)
    y = resample_poly(frames, CREPE_SR // g, int(sr) // g, axis=1)
    if y.shape[1] < CREPE_WINDOW:
        y = np.pad(y, ((0, 0), (0, CREPE_WINDOW - y.shape[1])))
    s = (y.shape[1] - CREPE_WINDOW) // 2
    w = y[:, s:s + CREPE_WINDOW]
    w = w - w.mean(1, keepdims=True)
    return w / np.maximum(w.std(1, keepdims=True), 1e-10)


def crepe_probabilities(frames: np.ndarray, sr: int, batch: int = 512) -> np.ndarray:
    """(n, 360) pitch-bin probabilities for each frame."""
    import torch

    model = _Crepe.get()
    w = crepe_windows(frames, sr).astype(np.float32)
    out = []
    for i in range(0, len(w), batch):
        out.append(model(torch.from_numpy(w[i:i + batch]).to(_Crepe.device)).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 360), np.float32)


def decode(probs: np.ndarray, fmin: float = DEFAULT_FMIN, fmax: float = DEFAULT_FMAX):
    """(n, 360) probabilities → (f0 Hz, raw peak probability), CREPE's local weighted average."""
    probs = np.atleast_2d(probs)
    p = np.where((_HZ >= fmin) & (_HZ <= fmax), probs, 0.0)
    k = np.argmax(p, axis=1)
    peak = p[np.arange(len(p)), k]
    f0 = np.zeros(len(p))
    for i, c in enumerate(k):
        lo, hi = max(c - 4, 0), min(c + 5, 360)
        wts = p[i, lo:hi]
        if wts.sum() > 0:
            f0[i] = 10.0 * 2.0 ** (np.sum(wts * _CENTS[lo:hi]) / wts.sum() / 1200.0)
    return f0, peak


class CrepePitch(Representation):
    """Pretrained CREPE as a pitch lens."""

    name = "crepe"
    medium = "audio"
    assumptions = ("the frame looks like the monophonic pitched training data (MIR-1K, Bach10, RWC-synth, "
                   "MedleyDB, MDB-stem-synth, NSynth)",
                   "the pitch is stable over the centre 64 ms",
                   "a normalised waveform carries the pitch (level cues are removed)")
    calibration = CALIBRATION

    def read(self, signal: np.ndarray, context: Mapping[str, Any]) -> Reading:
        f0, raw = self.read_batch(signal[None, :], context)
        return Reading(self.name, float(f0[0]) if f0[0] > 0 else None, float(raw[1][0]), {"raw": float(raw[0][0])})

    def read_batch(self, frames: np.ndarray, context: Mapping[str, Any]):
        """Many frames at once: (f0 array, (raw peak, confidence))."""
        probs = crepe_probabilities(frames, context["sr"])
        f0, peak = decode(probs, context.get("fmin", DEFAULT_FMIN), context.get("fmax", DEFAULT_FMAX))
        return f0, (peak, _scale(peak, *self.calibration))
