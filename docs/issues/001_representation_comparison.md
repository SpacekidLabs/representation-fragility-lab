# Issue #1: Compare fragility curves across representations under additive noise

## Goal
Determine whether STFT (Short-Time Fourier Transform), ACF (Autocorrelation Function), and Cepstrum exhibit distinct degradation profiles under controlled additive Gaussian noise.

## Description
To establish a baseline understanding of representation fragility, we want to measure and analyze how different signal representations degrade under additive noise. The current setup evaluates a 440 Hz sine wave under varying standard deviations of Gaussian noise (from 0.0 to 1.0).

Specifically, we want to:
1. Plot the fragility curves for STFT, ACF, and Real Cepstrum.
2. Compare the rate of decay of representation similarity (using Cosine Similarity).
3. Identify which representation is the most robust and which is the most fragile under additive noise.

## Current Findings (Summary)
* **STFT**: Exhibits a smooth, gradual decay under noise, dropping to around `0.57` similarity at `1.0` noise.
* **ACF**: Extremely robust under additive white noise (due to noise correlation properties), maintaining high similarity (`~0.95+` similarity) across the entire noise range.
* **Cepstrum**: Highly fragile. Similarity drops extremely fast as noise increases.

## Tasks
- [x] Implement STFT magnitude spectrogram representation (`src/representations/stft.py`)
- [x] Implement Fast Fourier Transform-based Autocorrelation representation (`src/representations/acf.py`)
- [x] Implement Real Cepstrum representation (`src/representations/cepstrum.py`)
- [x] Create automated experiments to run noise sweeps for each representation and generate comparison graphs.
- [x] Push curves to the repository.
