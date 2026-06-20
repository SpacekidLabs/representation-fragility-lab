# Issue #8: Failure Mechanism Identification

## Goal
For every major representation fragility mapped in the lab, identify the underlying mathematical operation responsible and outline targeted probing signals designed to isolate and excite these specific failure mechanisms.

---

## Mapped Failure Mechanisms

### 1. Autocorrelation (ACF) ↔ Period Relocation (Lag-Shift Sensitivity)
* **Responsible Operation**: Time-domain translation evaluation $\sum_n x[n]x[n+\tau]$ matched via rigid element-wise similarity.
* **Failure Mechanism**: Pitch shifting shifts the fundamental period $T_0 = 1/f_0$. This translates the autocorrelation peaks along the lag axis ($\tau$). Because similarity metrics (like Cosine or Pearson) compare vectors element-wise without translation invariance, any peak translation is treated as a complete decorrelation, causing a catastrophic drop in similarity.
* **Targeted Probing Signal**:
  - *Construction*: A signal whose fundamental frequency shifts by a fraction of a semitone (e.g., $10$ cents).
  - *Expected Behavior*: This will catastrophically drop ACF similarity (due to sub-sample/sample peak relocation) while leaving the STFT envelope virtually unaffected.

### 2. Real Cepstrum ↔ Log-Spectrum Instability (Noise-Floor Amplification)
* **Responsible Operation**: Logarithmic compression $\log(|X(f)| + \epsilon)$ applied to quiet frequency bins.
* **Failure Mechanism**: The derivative of the log function, $\frac{d}{dx}\log(x) = \frac{1}{x}$, approaches infinity as $x \to 0$. Consequently, tiny perturbations (such as additive noise) added to silent frequency regions are amplified by orders of magnitude, flooding the Cepstrum representation with random noise-floor variance and shifting the DC offset ($c_0$).
* **Targeted Probing Signal**:
  - *Construction*: A sparse harmonic signal with high amplitude peaks and wide, perfectly silent spectral gaps, perturbed by extremely low-amplitude noise (e.g., std dev = $0.001$).
  - *Expected Behavior*: The noise is too quiet to affect the waveform or the STFT peaks, but it will completely destroy Cepstrum similarity.

### 3. STFT Magnitude ↔ Frequency-Bin Displacement (Spectral Leakage)
* **Responsible Operation**: Discrete frequency bin indexing $X[k]$.
* **Failure Mechanism**: When a frequency peak shifts from exactly on-bin (matching the DFT grid) to off-bin, it leaks energy across multiple adjacent bins (spectral leakage), redistributing the shape. Furthermore, if a peak shifts by even one full bin, element-wise metrics treat the bins as orthogonal dimensions, dropping similarity.
* **Targeted Probing Signal**:
  - *Construction*: A pure sine wave shifted from $430.66$ Hz (exactly on a DFT bin center for $N=1024$ at $22050$ Hz) to $441.43$ Hz (exactly between two bin centers).
  - *Expected Behavior*: This will trigger spectral leakage and bin displacement in the STFT, causing similarity to drop, while ACF (which is phase-independent and continuous in time) remains stable.
