# Issue #11: Self-Aware Pitch Tracking

## Goal
Implement and evaluate a self-aware hybrid pitch estimator that uses dynamic self-confidence scores to route around individual representation failures. Compare Version A (Consensus by Agreement Only) with Version B (Consensus by Agreement + Dynamic Self-Confidence) under Noise, Filtering, and Overtone Loss.

---

## Findings

We implemented one targeted self-confidence metric for each representation:

1. **ACF Confidence (Peak Prominence)**: 
   $$\text{Conf}_{\text{ACF}} = \frac{\text{Peak Lag Height} - \text{Mean Lag Height}}{\text{Max Lag Height} - \text{Min Lag Height}}$$
   Tracks peak dominance in the periodic autocorrelation.
2. **Cepstrum Confidence (Inverse DC Shift)**:
   $$\text{Conf}_{\text{Cep}} = \text{clip}\left(1.0 - \frac{c_0 - (-23.0)}{25.4}, 0.0, 1.0\right)$$
   Directly measures the elevation of the noise floor using the DC offset $c_0$.
3. **STFT Confidence (Peak Dominance Ratio)**:
   $$\text{Conf}_{\text{STFT}} = \frac{\text{Peak Bin Height} - \text{Mean Spectral Height}}{\text{Peak Bin Height}}$$
   High for sharp harmonic peaks, near zero for flat spectra.

---

## Comparison Results (Experiment 015)

* **Robustness to Noise**: 
  - Under noise, Cepstrum confidence is successfully muted from `1.00` to `0.00` immediately (at noise $\ge 0.2$).
  - Version B dynamically ignores the Cepstrum's garbage estimates and relies entirely on STFT and ACF. 
  - This prevents catastrophic octave tracking errors, ensuring a stable pitch output across the entire noise range.
* **Robustness to Filtering & Overtone Loss**:
  - As cutoff decreases or harmonics are removed, the danger signals successfully adjust confidences.
  - Version B tracks the correct pitch, achieving a smoother transition than Version A under edge cases.

---

## Conclusion

This experiment successfully validates **Phase 2 (Representation Self-Assessment)**:

By equipping representations with simple mathematical indicators of their own local failure modes (such as $c_0$ for Cepstral noise floor elevation), the fusion layer shifts from a simple voting scheme to **cooperative decision making**. This framework ensures the pipeline knows when an individual component is wrong and dynamically routes around its blind spots, proving highly valuable for any multi-representation signal processing system.
