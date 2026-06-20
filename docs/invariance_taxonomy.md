# Invariance Taxonomy of Audio Representations

Based on the empirical findings in Experiments 001–008, we can classify audio representations by the invariants they preserve and the perturbations they ignore.

---

| Representation | Key Mathematical Invariants | Perturbations Ignored (Robustness) | Critical Fragilities |
| :--- | :--- | :--- | :--- |
| **STFT Magnitude** | Temporal shift invariance (within analysis window size). Discards phase. | Micro-temporal shift. | **Almost everything.** Additive noise, filtering, pitch shifting, and harmonic removal all directly alter bin magnitudes. |
| **Autocorrelation (ACF)** | Phase invariance (Fourier transform of power spectrum). | **Broadband additive noise.** Noise correlation decays to zero at non-zero lags over a long integration window. | **Pitch shifting.** Period changes cause linear shifts in lag-peaks, breaking vector alignment. |
| **Real Cepstrum** | Homomorphic filtering invariance ($\log(X \cdot H) = \log X + \log H$). | **Low-pass filtering** and **Harmonic/Overtone removal**. Preserves the fundamental periodicity structure. | **Additive noise.** Logarithmic scaling ($\log(|X| + \epsilon)$) dramatically amplifies the noise floor in quiet bins. |

---

## Detailed Invariance Proofs & Observations

### 1. Autocorrelation Noise Invariance
For a signal $s(t) = x(t) + n(t)$ where $n(t)$ is white Gaussian noise uncorrelated with $x(t)$:
$$R_{ss}(\tau) = R_{xx}(\tau) + R_{nn}(\tau) + R_{xn}(\tau) + R_{nx}(\tau)$$
Since $n(t)$ is white noise and uncorrelated with $x$:
$$R_{nn}(\tau) \approx 0 \quad (\text{for } \tau \neq 0)$$
$$R_{xn}(\tau) \approx 0$$
Therefore, for non-zero lags, $R_{ss}(\tau) \approx R_{xx}(\tau)$. The representation shape remains highly stable.

### 2. Cepstral Filter Invariance (Homomorphic Deconvolution)
For a signal $y(t) = x(t) * h(t)$ where $h(t)$ is a linear filter (like a low-pass filter):
$$|Y(f)| = |X(f)| \cdot |H(f)|$$
Taking the logarithm converts multiplication to addition:
$$\log |Y(f)| = \log |X(f)| + \log |H(f)|$$
Taking the inverse Fourier transform (cepstrum):
$$c_y(q) = c_x(q) + c_h(q)$$
Since the filter $h(t)$ is smooth, its cepstral energy $c_h(q)$ is concentrated at very low quefrencies (envelope). The high-quefrency structure (fine harmonic spacing $c_x(q)$) remains intact, explaining why Cepstrum cosine similarity is highly invariant to low-pass filtering and individual overtone removal.
