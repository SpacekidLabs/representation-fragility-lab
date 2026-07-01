"""
Physical descriptor extraction for the runtime framework.

This module keeps the public framework API independent from research
experiment modules, which may configure plotting, mutate sys.path, or load
calibration-only helpers at import time.
"""

from __future__ import annotations

import numpy as np
import librosa


def _compute_acf(frame: np.ndarray) -> np.ndarray:
    n = len(frame)
    spectrum = np.fft.rfft(frame, n=2 * n)
    acf = np.fft.irfft(spectrum * np.conj(spectrum), n=2 * n)
    return acf[:n]


def _compute_stft_magnitude(frame: np.ndarray) -> np.ndarray:
    magnitude = np.abs(librosa.stft(frame))
    if magnitude.ndim > 1:
        return np.mean(magnitude, axis=1)
    return magnitude


def extract_physical_descriptors(
    frame: np.ndarray,
    sr: int,
    n_fft: int = 2048,
) -> list[float]:
    """Extract the 10 physical descriptors used by the framework weights."""
    acf = _compute_acf(frame)
    magnitude = _compute_stft_magnitude(frame)

    mag_sum = np.sum(magnitude)
    if mag_sum > 1e-12:
        probabilities = np.clip(magnitude / mag_sum, 1e-12, 1.0)
        entropy = float(
            -np.sum(probabilities * np.log2(probabilities))
            / np.log2(len(probabilities))
        )
        log_mean = np.mean(np.log(magnitude + 1e-12))
        flatness = float(np.exp(log_mean) / (np.mean(magnitude) + 1e-12))
    else:
        entropy = 1.0
        flatness = 1.0

    zcr = float(np.mean(np.abs(np.diff(np.sign(frame)))) / 2.0)

    min_lag = int(sr / 1000)
    max_lag = min(int(sr / 80), len(acf))
    acf_range = acf[min_lag:max_lag]
    if acf_range.size:
        lag_idx = int(np.argmax(acf_range) + min_lag)
        harmonic_ratio = float(acf[lag_idx] / (acf[0] + 1e-12))
        periodicity = float(
            np.clip(
                (acf[lag_idx] - np.mean(acf_range))
                / (np.max(acf_range) - np.min(acf_range) + 1e-10),
                0.0,
                1.0,
            )
        )
    else:
        harmonic_ratio = 0.0
        periodicity = 0.0

    rms = np.sqrt(np.mean(frame**2))
    crest_factor = float(np.max(np.abs(frame)) / (rms + 1e-12))

    freqs = np.fft.rfftfreq(n_fft, d=1 / sr)
    cumulative = np.cumsum(magnitude)
    total = cumulative[-1]
    if total > 1e-12:
        idx = int(np.where(cumulative >= 0.85 * total)[0][0])
        rolloff = float(freqs[idx])
    else:
        rolloff = 0.0

    n_bins = len(magnitude)
    l1_norm = np.sum(np.abs(magnitude))
    l2_norm = np.sqrt(np.sum(magnitude**2))
    if l2_norm > 1e-12:
        ratio = l1_norm / l2_norm
        sparsity = float((np.sqrt(n_bins) - ratio) / (np.sqrt(n_bins) - 1.0 + 1e-12))
    else:
        sparsity = 0.0

    mean = np.mean(frame)
    std = np.std(frame) + 1e-8
    transientness = float(np.mean((frame - mean) ** 4) / (std**4))

    rms_values = [np.sqrt(np.mean(subframe**2)) for subframe in np.array_split(frame, 4)]
    modulation = float(np.std(rms_values))

    return [
        entropy,
        flatness,
        zcr,
        harmonic_ratio,
        crest_factor,
        periodicity,
        rolloff,
        sparsity,
        transientness,
        modulation,
    ]
