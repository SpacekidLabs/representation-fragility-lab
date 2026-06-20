# Issue #7: Metric Fragility & The Three-Body Problem

## Goal
Determine whether the observed fragility of representations (specifically Cepstrum under additive noise) is a fundamental limit of the representation itself or an artifact of the similarity metric used (Cosine Similarity vs. Pearson Correlation vs. Euclidean Distance).

---

## Findings

### 1. The Cepstrum Paradox Resolved
We compared Cosine Similarity, Pearson Correlation, and Euclidean Similarity on the Harmonic Stack representation:

* **Cosine Similarity**: Drops to `0.0000` immediately under noise ($0.1 \to 1.0$).
* **Pearson Correlation**: Becomes highly negative (`-0.96` to `-0.99`).
* **Euclidean Similarity**: Stabilizes at a low value of `0.33`.

### 2. The DC Component ($c_0$) Dominance
The correlation collapse is mathematically driven by the first cepstral coefficient ($c_0$), which represents the DC offset (average energy of the log-spectrum):
- In the clean signal, silent frequency bins are near zero ($\log(1e-10) \approx -23$), pulling the log-spectrum average ($c_0$) to a highly negative value (`-22.9`).
- In the noisy signal, the noise floor rises to around $+2.0$, pushing the log-spectrum average ($c_0$) to a positive value (`+2.4`).
- Mean-centering (Pearson Correlation) does not remove this offset because $c_0$ is the only large component in the cepstral vector. The dot product is dominated entirely by $c_0 \cdot c'_0$, which is negative, driving the correlation to near `-1.0`.

### 3. Truncating the DC Component
Even if we discard the DC component ($c_0$) and compute similarity on coefficients $c_1$ and above:
- **Cosine Similarity (no DC)**: `0.179` under $0.1$ noise.
- **Pearson Correlation (no DC)**: `0.179` under $0.1$ noise.
- **Euclidean Similarity (no DC)**: `0.023` under $0.1$ noise.

*Conclusion*: Discarding the DC component prevents the correlation from flipping negative, but the similarity remains extremely low. The logarithmic compression ($\log(|X| + \epsilon)$) amplifies random noise floor fluctuations across all silent bins, causing noise to dominate the variance of the higher-order cepstral coefficients as well. The fragility is indeed **intrinsic** to the combination of the Cepstrum representation and noise perturbation.

---

## The Three-Body Interaction

The experiment confirms that the measurement pipeline is a three-body problem:

1. **STFT** behaves identically under Cosine Similarity and Pearson Correlation because it is non-negative and lacks a massive single-coefficient bias.
2. **ACF** is highly robust across all three metrics.
3. **Cepstrum** is highly metric-dependent, flipping from `0.0` (Cosine) to `-0.96` (Pearson) due to the DC coefficient sign flip.
