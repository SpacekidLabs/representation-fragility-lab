"""Exp042: Mel-spectrogram and Chroma blind-spot atlas.

Extends the representation-fragility methodology to Mel-spectrogram and
Chroma representations by reusing the existing signal atlas, noise/filter
sweeps, and similarity metrics established in earlier experiments.

Reference methodology:
- src/experiments/exp001_noise_fragility.py
- src/experiments/exp003_compare_stft_vs_acf.py
- src/experiments/exp013_automated_discovery.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "results" / "exp042_mel_chroma_blind_spots"


def _hash_label(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]


def generate_signal(name: str, sr: int = 22050, duration: float = 1.0) -> np.ndarray:
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    if name == "sine":
        return np.sin(2.0 * np.pi * 440.0 * t)
    if name == "harmonic_stack":
        return sum(np.sin(2.0 * np.pi * f * t) for f in [220, 440, 880])
    if name == "fm_tone":
        carrier = 440.0
        modulator = 5.0
        index = 2.0
        return np.sin(2.0 * np.pi * carrier * t + index * np.sin(2.0 * np.pi * modulator * t))
    if name == "chirp":
        f0, f1 = 220.0, 880.0
        return librosa.chirp(fmin=f0, fmax=f1, sr=sr, duration=duration)
    if name == "percussive":
        click = np.zeros_like(t)
        click[:: max(1, int(sr * 0.05))] = 1.0
        return click
    if name == "noise":
        return np.random.default_rng(0).standard_normal(size=t.shape)
    raise ValueError(f"Unknown signal: {name}")


def apply_noise(x: np.ndarray, snr_db: float) -> np.ndarray:
    signal_power = np.mean(x ** 2)
    noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    noise = np.sqrt(noise_power) * np.random.default_rng(1).standard_normal(size=x.shape)
    return x + noise


def mel_representation(x: np.ndarray, sr: int) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=x,
        sr=sr,
        n_fft=2048,
        hop_length=512,
        n_mels=64,
        fmin=50,
        fmax=8000,
        power=2.0,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db


def chroma_representation(x: np.ndarray, sr: int) -> np.ndarray:
    chroma = librosa.feature.chroma_stft(
        y=x,
        sr=sr,
        n_fft=2048,
        hop_length=512,
        tuning=0.0,
        norm=2,
    )
    return chroma


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    if np.linalg.norm(a) == 0.0 or np.linalg.norm(b) == 0.0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def run_sweep(signal_name: str, sr: int = 22050, save: bool = False) -> None:
    clean = generate_signal(signal_name, sr=sr)
    mel_clean = mel_representation(clean, sr)
    chroma_clean = chroma_representation(clean, sr)

    snrs = [np.inf, 20.0, 10.0, 5.0, 0.0, -5.0]
    rows = []

    for snr in snrs:
        noisy = apply_noise(clean, snr)
        mel_noisy = mel_representation(noisy, sr)
        chroma_noisy = chroma_representation(noisy, sr)

        rows.append(
            {
                "signal": signal_name,
                "snr_db": snr,
                "mel_similarity": cosine_similarity(mel_clean, mel_noisy),
                "chroma_similarity": cosine_similarity(chroma_clean, chroma_noisy),
            }
        )

    frame = [row["snr_db"] for row in rows]
    plt.style.use("dark_background")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)

    axes[0].plot(frame, [row["mel_similarity"] for row in rows], marker="o", color="#ff2d55")
    axes[0].set_title(f"Mel Similarity vs SNR: {signal_name}")
    axes[0].set_xlabel("SNR dB")
    axes[0].set_ylabel("Cosine Similarity")
    axes[0].set_ylim(-0.2, 1.05)
    axes[0].grid(True, linewidth=0.4)

    axes[1].plot(frame, [row["chroma_similarity"] for row in rows], marker="o", color="#30d158")
    axes[1].set_title(f"Chroma Similarity vs SNR: {signal_name}")
    axes[1].set_xlabel("SNR dB")
    axes[1].set_ylabel("Cosine Similarity")
    axes[1].set_ylim(-0.2, 1.05)
    axes[1].grid(True, linewidth=0.4)

    if save:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        fig_path = OUTPUT_DIR / f"exp042_{signal_name}_snr_sweep.png"
        json_path = OUTPUT_DIR / f"exp042_{signal_name}_snr_sweep.json"
        fig.savefig(fig_path, dpi=160)
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2)
        print(f"Saved: {fig_path}")
        print(f"Saved: {json_path}")

    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", action="store_true", help="Save plots and JSON artifacts")
    parser.add_argument("--signal", default="harmonic_stack", help="Signal class to sweep")
    parser.add_argument("--sr", type=int, default=22050, help="Sample rate")
    args = parser.parse_args()

    run_sweep(args.signal, sr=args.sr, save=args.save)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
