# Issue #11: Representation Confidence Modeling (Phase 2)

## Goal
Transition from passive mapping of representation blind spots to active confidence modeling. Equip each representation (STFT, ACF, Cepstrum) with the mathematical capacity to recognize when it is approaching its own known blind spot and report a dynamic self-confidence score.

---

## Representation-Specific Danger Signals

To teach representations when to distrust themselves, we will model three distinct classes of structural indicators:

### 1. Autocorrelation (ACF) Danger Signals
- **Low Peak Contrast (Prominence)**: The height of the primary lag peak $R(\tau_{\text{peak}})$ relative to the mean of $R(\tau)$. If the peak is flat, noise dominates.
- **Multi-Peak/Octave Ambiguity**: The presence of multiple peaks of almost identical height at integer multiples of the period (e.g. $\tau, 2\tau, 3\tau$). This indicates octave confusion.
- **Lag Peak Instability**: High temporal variance of $\tau_{\text{peak}}$ across short overlapping frames.

### 2. Cepstrum Danger Signals
- **Noise-Floor Offset ($c_0$) Shift**: Rapid shift in the log-spectral average offset. An offset close to positive values indicates the presence of an elevated noise floor.
- **Low-Quefrency Saturation**: High concentration of energy in very low quefrencies ($c_1$ to $c_{10}$). This indicates dominant low-pass filtering or spectral tilt that is masking fine harmonic details.
- **Lack of Harmonic Support**: Absence of secondary cepstral peaks at integer multiples/divisors of the primary quefrency period.

### 3. STFT Danger Signals
- **High Spectral Entropy**: If the average spectrum is flat (high entropy), the signal is noise-like rather than tone-like.
- **Spectral Leakage Ratio**: High energy in bins adjacent to the peak relative to the peak itself ($X[k \pm 1] / X[k]$). Indicates off-bin leakage.
- **Weak Peak Dominance**: Small ratio of the peak bin magnitude to the total sum of the spectrum.

---

## Fusion Architecture: Confidence-Weighted Voting

In Phase 2, the consensus scoring function will be upgraded from a flat self-reported peak amplitude to a multi-factor confidence model:

$$C_i = f(\text{Danger Signals}_i)$$

The consensus scoring kernel will then weight candidate pitch values by this dynamic confidence score:
$$\text{Score}(p_j) = \sum_i C_i \cdot \exp\left(-\frac{|p_j - p_i|}{\sigma}\right)$$

This forces the fusion layer to dynamically mute representations that are self-reporting proximity to a blind spot (e.g., muting Cepstrum when noise is high, or muting ACF when pitch shifts rapidly).
