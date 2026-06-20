# Issue #6: Representation Invariance Across Signal Classes

## Goal
Determine whether the invariants discovered so far (e.g., ACF noise-invariance, Cepstrum filter-invariance) are universal properties of the representations themselves or if they are highly specific to harmonic periodic signals.

---

## Proposed Signal Classes

To rigorously test universality, we will evaluate representations against six distinct signal classes:

1. **Sine**: A single pure frequency (no overtones). Evaluates baseline representation sensitivity.
2. **Harmonic Stack**: Fundamental + integer multiple overtones. Evaluates harmonic structure and pitch-related invariants.
3. **FM Tone (Frequency Modulation)**: A carrier frequency modulated by a modulator. Evaluates behavior under distributed, non-integer sideband spectral peaks.
4. **Chirp (Frequency Sweep)**: A signal whose frequency increases or decreases linearly/exponentially over time. Evaluates non-stationary signals.
5. **Percussive Impulse**: A short, broadband impulse (transient). Evaluates localized temporal events vs. stationary representations.
6. **Noise (Gaussian White)**: Stochastic broadband signal with no periodic structure.

---

## Evaluation Grid

We will measure three key dimensions for each combination:

| Dimension | Description |
| :--- | :--- |
| **Fragility Profiles** | How rapidly the similarity curve decays under perturbations (Noise, Pitch, Time Stretch, Filter). |
| **Invariant Preservation** | Which symmetries (e.g., homomorphic filter addition, noise orthogonality) hold true for that specific signal class. |
| **Failure Signatures** | The visual and quantitative "shape" of degradation (e.g., catastrophic step-function drop vs. graceful asymptotic decay). |

---

## Expected Research Questions

1. **Does ACF noise robustness hold for transient impulses?**
   - *Hypothesis*: Yes, because the autocorrelation of a single impulse is still an impulse, and white noise averages out.
2. **Does Cepstral filter-invariance hold for non-stationary chirps?**
   - *Hypothesis*: Real Cepstrum discards time-localization. A filtered chirp will still show envelope similarity in the cepstral domain, but time-varying frequency details are lost globally.
3. **How does STFT compare to ACF/Cepstrum on stochastic noise?**
   - *Hypothesis*: On noise, ACF remains highly invariant (peaks at zero), while STFT magnitude similarity decays immediately as the random spectral realization shifts.
