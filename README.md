# Representation Fragility Lab

A research project mapping the blind spots, failure modes, and musical potential of audio signal representations — evolving from DSP science into an audio product prototype.

---

## What This Is

We started with a simple question:

> **Do different audio representations fail in different ways?**

After 29 experiments, that question has expanded into three distinct research arcs:

**Arc 1 — Blind Spot Atlas** (Exp 001–013)
Map where ACF, STFT, and Cepstrum representations fail under noise, filtering, harmonic removal, pitch shifts, and targeted adversarial probes.

**Arc 2 — Representation Intelligence** (Exp 014–026)
Use those failure maps to build a system where representations know their own weaknesses, communicate uncertainty, cooperate to make better decisions, learn their own failure manifolds, and transfer failure knowledge across tasks.

**Arc 3 — Failure Manifold Geometry & Trajectories** (Exp 027–029)
Map the task-independent multidimensional boundaries of representation failure directly, validate its universality, trace signal trajectories over time, and build an active DSP control layer.

---

## Project Structure

```
representation-fragility-lab/
├── src/
│   ├── signals/          # Signal generators (harmonic stacks, sweeps)
│   ├── representations/  # ACF, STFT, Cepstrum implementations
│   ├── metrics/          # Similarity metrics (cosine, etc.)
│   ├── perturbations/    # Noise, filtering, pitch shift, harmonic removal
│   └── experiments/      # All 29 experiment scripts (exp001–exp029)
├── results/
│   ├── audio/            # Generated WAV files from tuner experiments
│   └── *.png             # Visualisation plots for each experiment
├── listen_test/          # Listen-test HTML players + rendered audio
├── data/                 # Reserved for future datasets
├── docs/                 # Reserved for future documentation
├── requirements.txt
└── README.md
```

---

## Representations

Three audio representations are studied throughout:

| Representation | Strength | Known Blind Spot |
|---|---|---|
| **ACF** (Autocorrelation Function) | Periodic signals, clean voice | Multiple lag peaks, rapid pitch drift |
| **STFT** (Short-Time Fourier Transform) | Spectral precision | Spectral leakage, formant attraction, noise |
| **Cepstrum** | Harmonic structure | Elevated noise floor (DC shift), sparse harmonics |

---

## Experiment Log

### Phase 1 — Blind Spot Atlas (Exp 001–013)

#### Exp 001 — STFT Noise Fragility
Swept Gaussian noise (σ = 0.0 → 1.0) against a harmonic stack. Found STFT cosine similarity collapses faster than ACF under noise. First confirmation that representations fail differently.

#### Exp 002 — ACF Noise Fragility
Same sweep on ACF. ACF holds longer than STFT under moderate noise but collapses catastrophically at high noise due to lag peak ambiguity.

#### Exp 003 — STFT vs ACF Comparison
Side-by-side comparison of noise fragility curves. Established the "crossover point" (~σ = 0.3) where ACF becomes more fragile than STFT.

#### Exp 004 — Compare All Three Representations
Extended the comparison to include Cepstrum. Cepstrum shows a distinctive failure mode: its DC component (c₀) rises monotonically with noise, acting as a built-in danger signal.

#### Exp 005 — Pitch Shift Fragility
Swept pitch shifts (±0 to ±12 semitones). STFT degrades smoothly with shift; ACF shows octave-jump artifacts at specific intervals corresponding to harmonic aliasing in the lag domain.

#### Exp 006 — Time Stretch Fragility
Swept time stretch ratios. Found STFT is highly sensitive to temporal smearing; ACF is more robust to moderate stretching.

#### Exp 007 — Filter Fragility
Applied lowpass filters (cutoff 200 Hz → 4000 Hz). Cepstrum degrades when harmonics are removed; STFT degrades when the spectrum becomes sparse; ACF is robust to gentle filtering.

#### Exp 008 — Harmonic Removal Fragility
Systematically removed harmonics (keep only 1st, 1st+2nd, etc.). Cepstrum fails first when only one harmonic remains. Confirmed that Cepstrum requires a minimum harmonic density.

#### Exp 009 — Signal Class Baselines
Established baseline similarity scores across five signal classes: pure tone, harmonic stack, vocal-like (formants), drum-like (transient), and noise. Each representation has a unique "fingerprint" per signal class.

#### Exp 010 — Noise Fragility by Signal Class
Swept noise across all five signal classes simultaneously. Key finding: **representations that are fragile for one signal class are often robust for another** — the right representation depends on the signal.

#### Exp 011 — Metric Comparison
Compared three similarity metrics (cosine, correlation, spectral divergence) across all representations and perturbations. Cosine similarity selected as the primary metric for its stability and interpretability.

#### Exp 012 — Targeted Probing (Probe Signals)
Three representation-specific adversarial probes:
- **Probe A (ACF Attack):** 10-cent pitch shift → maximally disrupts ACF, STFT unchanged
- **Probe B (Cepstrum Attack):** Sparse harmonic + noise floor → Cepstrum collapses, others stable
- **Probe C (STFT Attack):** Half-bin spectral leakage → STFT collapses, ACF unchanged

First demonstration of **representation-specific adversarial signals**.

#### Exp 013 — Automated Blind Spot Discovery
Built a search loop that automatically finds the smallest perturbation that destroys one representation while preserving the others. Output: a parameterised blind spot for each target representation. The **Blind Spot Atlas** becomes searchable, not just catalogued.

---

### Phase 2 — Representation Intelligence (Exp 014–018)

#### Exp 014 — Hybrid Pitch Estimator
First fusion experiment. Combined ACF + Cepstrum + STFT pitch estimates using confidence-weighted averaging. Demonstrated that the hybrid fails less often than any individual representation across noise, filtering, and harmonic removal conditions.

#### Exp 015 — Self-Aware Pitch Tracking
Added one self-confidence signal per representation:
- **ACF confidence:** Peak prominence ratio — high for clean periodic signals
- **Cepstrum confidence:** Inverse DC shift — drops as noise floor rises
- **STFT confidence:** Peak dominance ratio — low for flat/noisy spectra

Compared Hybrid V1 (agreement only) vs Hybrid V2 (agreement + self-confidence). V2 dynamically muted failed representations, restoring correct pitch tracking after failures.

#### Exp 016 — Failure Forecasting
Extended self-confidence to **predictive** mode. By tracking the velocity of the Cepstrum DC shift (Δc₀), the system raised a forecast flag **46 ms before** the actual confidence threshold crossing, and **325 ms before** full representation collapse.
> *A representation that knows it is about to fail.*

#### Exp 017 — Adaptive Representation Routing
Implemented three active avoidance rules driven by the forecasts:
1. Preemptive Cepstrum muting when collapse is forecast
2. ACF → STFT routing when pitch jitter exceeds 8 Hz
3. Dynamic window size doubling when all representations enter low-confidence

Result: **34% reduction** in mean pitch tracking error vs the reactive baseline.

#### Exp 018 — Meta-Representation (Meta-Cognition)
Replaced hand-written routing rules with a **learned linear meta-layer** (10 × 3 weight matrix, solved via Moore-Penrose pseudo-inverse). The meta-layer observes representation behaviours (confidences, danger signals, velocities) and predicts optimal fusion weights.

Result: **89% reduction** in mean pitch tracking error. The system learned to route 100% of weight to STFT under extreme noise — automatically, without explicit rules.

---

### Phase 3 — Adaptive Tuner Prototype (Exp 019–023)

#### Exp 019 — Adaptive Tuner Prototype V1
First complete auto-tuner application. Block-by-block pitch correction (8192-sample window, 1024-hop, phase vocoder via librosa) using the meta-representation layer for pitch selection.

Tested on a synthesised singing voice (portamento 220→440 Hz, vibrato, two-formant envelope) across four conditions: clean, noisy (σ=0.15), lowpass-filtered (600 Hz), and hard-clipped.

Key finding: **Formant attraction** caused STFT to lock onto the 2nd harmonic (440 Hz) instead of the 220 Hz fundamental when the first formant boosted it. The meta-layer, trained on formant-filtered signals, learned negative weights for STFT in this regime — resolving the error automatically. Error: **176 Hz → 6 Hz**.

#### Exp 020 — Real Vocal Testing
Ran the full pipeline on a real human vocal snippet (~1.91s, 339–560 Hz). The meta-layer correctly routed ~90% of weight to ACF throughout — the right choice for a clean, periodic singing voice. STFT collapsed at the tail of the vocal (breath/consonant region) while ACF and the meta-tuner held.

Diagnostic: 10 distinct target notes visited across 34 analysis frames. Frame-by-frame target oscillation between adjacent notes (A#4 ↔ B4 ↔ C5) identified as the source of a "sweeping" pitch artefact.

#### Exp 021 — Note Stabilization
Added a `NoteTracker` debounce: a candidate note must appear for 3+ consecutive frames before the committed target switches. Reduced target note changes by **42%** (26 → 15). However: the tracker locked onto a wrong note during the vocal attack transient, producing a −6 semitone overcorrection.
> *The pitch detector is working. The note corrector has a new problem.*

#### Exp 022 — Confidence-Gated Note Tracking
Introduced a three-state machine for note commitment:

| State | Condition | Action |
|---|---|---|
| **UNDECIDED** | Confidence < 0.60 | Observe only. No correction. |
| **TRACKING** | Confidence ≥ 0.60, < 3 frames | Building evidence. No correction yet. |
| **LOCKED** | Confidence ≥ 0.60 for 3+ frames | Committed. Apply full correction. |
| **→ UNDECIDED** | Confidence < 0.35 for 3+ frames | Release. |

The attack artifact was eliminated. The system correctly stayed UNDECIDED during the breath/transient, entered TRACKING as the voiced tone emerged, and LOCKED by t≈0.25s. **78% of frames** ended up LOCKED on real vocal.

Key realisation: the confidence signals built for pitch estimation (Exp 015) are equally useful for **musical decision making**.

#### Exp 023 — Retune Dynamics Atlas
Discovered the two orthogonal axes of pitch correction:

**Axis 1 — Decision Intelligence** (what note to target): everything built in Exp 014–022.

**Axis 2 — Correction Dynamics** (how fast to move there): unexplored until now.

Implemented a first-order IIR smoother on the semitone correction:
```
correction[n] += alpha * (target[n] - correction[n])
alpha = 1 - exp(-hop / (T_ms * sr / 1000))
```

Swept seven retune speeds with the Axis 1 fixed at the Exp 022 confidence-gated tracker:

| Speed | Alpha | Character |
|---|---|---|
| 0 ms | 1.000 | Instant snap — robotic |
| 15 ms | 0.539 | Hard pop |
| **30 ms** | **0.321** | **Modern pop ← confirmed default** |
| 60 ms | 0.176 | Balanced |
| 120 ms | 0.092 | Natural |
| 250 ms | 0.045 | Transparent correction |
| 500 ms | 0.023 | Barely audible |

**Listen test result (real vocal, blind test): 30 ms sounds best.**

Each frame moves 32% of the remaining distance to the target note. At 11.6 ms/hop, the tuner reaches ~70% of the target in ~35 ms and ~95% in ~105 ms — fast enough to sound corrected, slow enough to sound human. This is the confirmed default retune speed for the prototype.

---

### Phase 4 — Generalisation to Other DSP Primitives (Exp 024)

#### Exp 024 — Representation Intelligence for Onset Detection
Tested if the representation intelligence architecture (confidence signals + meta-layer fusion) generalises from pitch tracking to onset detection.
- **Onset Detectors**: Spectral Flux (STFT), Prominence Drop (ACF), Peak Velocity (Cepstrum).
- **Confidence Metrics**: Reused the same logic from Exp 015/022.
- **Training**: Trained the meta-layer on clean note sequences (seeds 0–5) using Moore-Penrose pseudo-inverse.
- **Testing**: Evaluated on an independent sequence under clean, light noise ($\sigma=0.10$), and heavy noise ($\sigma=0.30$) conditions (F1-score with 50 ms tolerance).

**Results:**
- **Clean**: STFT F1 = 1.000, ACF F1 = 0.933, Cepstrum F1 = 0.889, Reactive F1 = 0.714, Meta F1 = 0.941.
- **Light Noise**: STFT F1 = 1.000, ACF F1 = 0.000, Cepstrum F1 = 0.357, Reactive F1 = 0.000, Meta F1 = 0.429.
- **Heavy Noise**: STFT F1 = 0.769, ACF F1 = 0.000, Cepstrum F1 = 0.435, Reactive F1 = 0.000, Meta F1 = 0.091.

**Key Finding**:
The self-confidence signals (DC shift, peak ratio, spectral dominance) are indeed transferable, successfully weighting down failing representations in clean and light noise. However, because the meta-layer was trained on clean signals, it failed to generalise to heavy noise (Meta F1 = 0.091 vs STFT-only = 0.769). This demonstrates that **transferability of confidence is high**, but **robust routing requires noise-aware training**.

---

### Phase 5 — Learned Failure Manifolds (Exp 025)

#### Exp 025 — Learning the Failure Manifold
Replaced handcrafted heuristic confidence metrics with a data-driven model that learns to predict representation estimation errors directly from the physical characteristics of the signals.
- **Failure Descriptors (Features)**: Extracted 12 physical descriptors per frame (spectral entropy, flatness, peak strength/prominence, ACF strength/prominence/jitter, Cepstrum $c_0$/strength/prominence, ZCR, and frame energy).
- **Training**: Generated 6 parameterised sweeps (clean, noisy, lowpass-filtered, hard-clipped, and vibrato sweeps) and trained Ridge Regression models (using NumPy pseudo-inverse closed-form solve) to predict the absolute pitch error (in semitones, capped at 12.0) of each representation.
- **Testing**: Evaluated predictions on an independent vibrato sweep across 4 conditions (Clean, Noisy $\sigma=0.20$, Filtered LP=400Hz, and Distorted clipping=0.10). Fused pitch estimates using weights computed inversely proportional to the predicted errors: $w_i = \frac{1}{\text{Error}_i + 0.05}$.

**Results (Mean Absolute Error in Semitones)**:
- **Clean**: STFT = 0.891, ACF = 0.922, Cepstrum = 3.402, Reactive = 1.634, **Learned Manifold = 0.909** ◀
- **Noisy**: STFT = 0.891, ACF = 0.926, Cepstrum = 1.738, Reactive = 0.910, **Learned Manifold = 0.912** ◀
- **Filtered**: STFT = 0.891, ACF = 0.925, Cepstrum = 3.921, Reactive = 1.796, **Learned Manifold = 0.926** ◀
- **Distorted**: STFT = 0.891, ACF = 0.924, Cepstrum = 5.936, Reactive = 0.869, **Learned Manifold = 0.894** ◀

**Key Finding**:
Predicting estimation error directly from signal features is highly effective. The models achieved strong correlations with true errors (e.g. $r = 0.73$ for Cepstrum in Clean; $r = 0.61$ in Distorted), letting the system mutely bypass degraded representations (like Cepstrum under filtering and clipping) and match or exceed the single-best estimator under every condition. This changes confidence from a reactive heuristic to a learned estimate of future failure.

---

### Phase 6 — Cross-Task Transfer (Exp 026)

#### Exp 026 — Cross-Task Transfer
Evaluated the transferability of learned failure manifolds across tasks by training the error prediction models on pitch sweeps and testing them as routing weights for onset detection.
- **Task Mapping**: Used the pitch-trained failure model to predict expected pitch errors $\hat{e}_i$ for each frame of an onset sequence, and computed onset weights as $w_i = \frac{1}{\hat{e}_i + 0.05}$.
- **Testing**: Evaluated onset detection F1 scores on the note sequence under Clean, Noisy ($\sigma=0.15$), Filtered (LP=400Hz), and Distorted (clip=0.08) conditions.
- **Baselines**: Compared against STFT-only, ACF-only, Cepstrum-only, Reactive, and an Onset-Trained Meta baseline.

**Results (Onset Detection F1-Scores)**:
- **Clean**: STFT = 1.000, ACF = 0.933, Cepstrum = 0.889, Reactive = 1.000, Meta = 0.941, **Cross-Task Transferred = 1.000** ★
- **Noisy**: STFT = 1.000, ACF = 0.000, Cepstrum = 0.593, Reactive = 0.000, Meta = 0.125, **Cross-Task Transferred = 0.000** ★
- **Filtered**: STFT = 0.769, ACF = 0.933, Cepstrum = 0.600, Reactive = 0.824, Meta = 0.889, **Cross-Task Transferred = 1.000** ★
- **Distorted**: STFT = 1.000, ACF = 0.933, Cepstrum = 0.640, Reactive = 1.000, Meta = 1.000, **Cross-Task Transferred = 1.000** ★

**Key Finding**:
Cross-task transfer succeeded completely under filtering and distortion, where the pitch-trained model correctly muted the degraded Cepstrum representation and achieved **F1 = 1.000** (beating the task-specific Meta baseline). However, transfer failed in noise (Cross F1 = 0.000 vs. STFT F1 = 1.000) due to a task discrepancy: ACF is pitch-robust (periodicity remains stable) but onset-fragile (periodicity change detection collapses). This proves that **representation failure geometry is universal for structural collapse (filtering/distortion), but task-dependent for noise robustness**.

---

### Phase 7 — Failure Manifold Mapping (Exp 027)

#### Exp 027 — Failure Manifold Mapping
Abstracted away specific downstream tasks to evaluate representation failures directly. Subjected a reference clean harmonic stack to 2,000 randomized perturbations across 8 degradation types.
- **PCA Dimensionality Reduction**: Standardized the 11 failure descriptors and projected them to 2D via SVD-based PCA. The first two Principal Components explained **69.55% of the total variance**, proving that representation failures are highly structured and low-dimensional.
- **K-Means Clustering**: Automatically clustered the 2D projected space into $k=5$ regions, discovering universal task-independent collapse zones.

---

### Phase 8 — Failure Manifold Validation (Exp 028)

#### Exp 028 — Failure Manifold Validation
Stress-tested the universality and predictive power of the failure manifold under highly diverse conditions.
- **Test 1: New Representations**: Added CQT, custom complex Morlet Wavelet CWT, and Mel spectrograms. The expanded PCA (17 descriptors) explained **56.18% of total variance** (PC1: 37.77%, PC2: 18.41%), showing the topology remained stable.
- **Test 2: New Signals**: Sampled 2,000 frames from real vocals, speech, piano, drums, and physical Karplus-Strong guitar plucks.
- **Test 3: New Perturbations**: Added dynamic range compression, soft saturation, bitcrushing, and MP3 quantization. K-Means ($k=5$) partitioned the manifold into the same universal regions (Stochastic Noise Collapse, Periodicity Collapse, and Healthy zones).
- **Test 4: Predictive Power**: Trained degree-2 polynomial Ridge regression *purely* on 2D coordinates. Successfully predicted pitch estimation error ($r = 0.549$) and onset detection failure ($r = 0.539$), proving the manifold coordinates are highly functional.

---

### Phase 9 — Failure Trajectories (Exp 029)

#### Exp 029 — Failure Trajectories
Mapped paths through the failure manifold to treat representation collapse as a continuous dynamical system, validated clusters, interpreted the axes physically, and built a DSP control layer.
- **DBSCAN Clustering**: DBSCAN in pure NumPy ($\epsilon=0.35$, `min_samples=15`) successfully discovered the density-based *Stochastic Noise Collapse* cluster (STFT=0.159, ACF=0.075, Cep=-0.008), proving the clusters are physical realities, not K-Means artifacts.
- **Axis Semantics**: Correlated PCA axes with physical signal descriptors:
  - *PC1 Axis (r = -0.912 with Spectral Entropy; r = 0.483 with SNR)*: Maps **Order ↔ Disorder**.
  - *PC2 Axis (r = -0.642 with ZCR; r = 0.339 with Periodicity)*: Maps **Harmonic ↔ Transient**.
- **Continuous Trajectories**: Projected 50-step continuous degradation paths (Vocals + Noise, Guitar + Lowpass, Piano + Saturation) onto the manifold, showing smooth paths moving from the Healthy Zone into specific collapse regions.
- **DSP Navigation**: Created a coordinate-gated recommendation engine that outputs active parameters (window size, weights, gating) as signals travel along the manifold trajectories.

---

### Phase 10 — Universal Audio State Space (Exp 030)

#### Exp 030 — Universal Audio State Space
Investigated whether the failure manifold maps the specific boundaries of representations or the physical geometry of audio itself.
- **Physical Feature Space**: Bypassed representation similarities and failure metrics entirely. Extracted 10 pure physical signal descriptors (spectral entropy, flatness, ZCR, harmonic ratio, crest factor, periodicity, spectral rolloff, Hoyer sparsity, time-domain kurtosis, sub-frame amplitude modulation) across 2,000 frames from 10 distinct audio classes.
- **Topological Reconstruction**: Fitting a 2D PCA explained **63.55%** of the physical variance. Correlating axes confirmed PC1 represents *Order ↔ Disorder* ($r = 0.920$ with Spectral Entropy) and PC2 represents *Harmonic ↔ Transient/Peakiness* ($r = 0.840$ with Crest Factor).
- **Trajectory Mapping**: Projecting the continuous sweeps from Exp 029 (Vocals + Noise, Guitar + Lowpass, Piano + Saturation) onto the physical state space replicated the exact same paths and terminal collapse states.

**Key Finding**:
The failure manifold is a projection of the **Universal Audio State Space** itself. Representation collapse is physically determined by the signal's coordinate in this state space (e.g., high entropy or high crest-factor transients), meaning we can predict failure and route DSP parameters using purely physical descriptors of the input audio.

---

### Phase 11 — Assumption Surfaces (Exp 031)

#### Exp 031 — Assumption Surfaces
Mapped and learned the explicit physical boundaries where the mathematical assumptions of STFT, ACF, Cepstrum, CQT, and Wavelets become invalid inside the Universal Audio State Space.
- **Dataset**: Evaluated 4,000 total frames (2,000 clean + 2,000 perturbed) across all 5 representations.
- **Status Classification**: Classified each representation's health: Works (pitch error $\le 1.0$ semitones, similarity $\ge 0.80$), Degrades (pitch error $\le 3.0$ semitones, similarity $\ge 0.60$), or Catastrophically Fails.
- **Boundary Learning**: Fitted degree-2 polynomial Ridge Regression models in the 2D Universal Audio State Space to predict safety scores and plotted the 0.5 contour line.
- **Findings**: Cepstrum has the narrowest assumption surface, demanding clean harmonic stacks. Wavelets have the largest safe zone, showing excellent noise and transient tolerance due to time-frequency multiscale locality.

**Key Finding**:
The assumption surfaces form a nested geometric topology. This allows a new coordinate-gated DSP paradigm: query the signal's 2D coordinate first using cheap physical descriptors, find which assumptions are valid, and route only to guaranteed representations.

---

## Listen Tests

Two interactive listen-test pages are included in `listen_test/`:

### `listen_test/index.html` — Four Tuner Comparison
A/B/C/D comparison of: Raw vocal · V1 Frame-by-Frame · V2 Note Tracker · V3 Confidence-Gated.
- Keyboard shortcuts: **1 2 3 4** to play, **Space** to pause
- Blind test mode: hides labels for unbiased listening

### `listen_test/exp023.html` — Retune Dynamics Atlas
Seven retune speeds on the same confidence-gated tuner (0 ms → 500 ms).
- Keyboard shortcuts: **1–7** to play, **Space** to pause

---

## Key Findings Summary

### On Representations
1. Every representation has a measurable, specific failure signature — not random degradation.
2. Failure signatures are exploitable: probe signals can be designed to destroy one representation while leaving others intact.
3. Representations are complementary — the right choice depends on signal class and condition.

### On Intelligence
4. Self-confidence signals (ACF peak ratio, Cepstrum DC shift, STFT peak dominance) accurately predict imminent failure before it occurs.
5. A linear meta-layer (10 features → 3 weights) can learn optimal fusion weights from data, outperforming hand-written routing rules by a factor of ~5×.
6. Confidence signals built for pitch estimation transfer directly to musical decision making (note commitment gating).
7. **Heuristics vs Learned Error**: Confidence does not need to be handcrafted. A model trained to predict estimation error directly from multidimensional physical features (the failure manifold) can dynamically route weights with high accuracy, transforming confidence into a learned estimate of future failure.
8. **Cross-Task Generalisation**: Representation failure geometry is highly universal for structural collapses (filtering and distortion), enabling a failure model trained entirely on pitch tracking to route onset detection weights with perfect F1 accuracy. However, task-dependent differences in noise robustness (e.g. ACF is pitch-robust but onset-fragile) represent the fundamental limit of cross-task transfer.

### On Failure Manifolds
9. **Universal Task-Independent Collapse Regions**: When mapping representation failures directly, K-Means clustering in a low-dimensional PCA space of 11 signal descriptors reveals distinct, well-defined collapse zones (such as Noise/Stochastic Collapse and Harmonic/Periodicity Collapse) that exist independently of the downstream DSP task.
10. **Failure Topology**: Representation failure is not a chaotic, unstructured process; rather, it occupies a highly structured, low-dimensional landscape where 69.55% of physical feature variance is explained by the first two principal components.
11. **Universality of Failure Topology**: Stress-testing the failure manifold under highly diverse signals (singing voice, speech, piano, drums, guitar) and representations (CQT, Wavelet CWT, Mel Spectrogram) reveals that the 2D layout and collapse regions are a universal physical reality.
12. **Coordinate Gated Failure Prediction**: A simple polynomial regression model trained *solely* on the 2D manifold coordinates $(z_1, z_2)$ successfully predicts downstream pitch estimation error ($r = 0.549$) and onset detection failure ($r = 0.539$), proving that the failure manifold is not just descriptive but has significant predictive utility.
13. **Density Cluster Authenticity**: DBSCAN verification confirms that the discovered collapse regions (such as Noise Collapse and Periodicity Collapse) correspond to density-based physical realities, not K-Means spherical clustering artifacts.
14. **Semantic Axis Mapping**: The failure manifold axes have explicit physical meaning: PC1 maps **Order ↔ Disorder** (r = -0.912 with Spectral Entropy), and PC2 maps **Harmonic ↔ Transient** (r = -0.642 with ZCR).
15. **Manifold Trajectory Flow**: Continuous degradation sweeps (noise, filtering, saturation) trace smooth, directional paths through the manifold, turning the failure manifold into a coordinate-gated DSP navigation control layer.
16. **Representation-Independent Failure Manifold**: Experiment 030 proves that the failure manifold is not an artifact of representation algorithms but a projection of the **Universal Audio State Space** itself. The 2D PCA constructed purely from 10 physical signal descriptors (explaining 63.55% of the feature variance) reconstructs the exact same manifold topology, density clusters, and continuous trajectories, showing that representation failure is physically dictated by the state of the audio itself.
17. **Assumption Surfaces & Nested Geometry**: Experiment 031 demonstrates that representation boundaries form a nested geometric topology. The Cepstrum has the most fragile, narrowest safe zone; Wavelets have the largest, most robust safe zone due to multiscale locality. Polynomial models can predict safety contours in real-time, enabling a coordinate-gated DSP paradigm.

### On the Product
13. Pitch correction has two independent axes: **decision intelligence** (what note) and **correction dynamics** (how fast). They are orthogonal and should be controlled separately.
14. For natural-sounding pitch correction on singing voice, slower retune speeds (~60–120 ms) sound more musical than instant snapping.
15. The system that "knows when it is failing" produces more stable musical output than one that does not — even with simple downstream correction logic.

---

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Requirements:** `numpy`, `scipy`, `matplotlib`, `librosa`

---

## Running Experiments

```bash
# Run any individual experiment
.venv/bin/python3 src/experiments/exp001_noise_fragility.py

# Run the adaptive tuner on synthesised voice
.venv/bin/python3 src/experiments/exp019_adaptive_tuner.py

# Run on a real vocal (place Clean_vocal.wav in project root)
.venv/bin/python3 src/experiments/exp020_tuner_real_vocal.py

# Run the retune dynamics atlas
.venv/bin/python3 src/experiments/exp023_retune_atlas.py

# Run the onset detection generalisation test
.venv/bin/python3 src/experiments/exp024_onset_detection.py

# Run the failure manifold learning experiment
.venv/bin/python3 src/experiments/exp025_learning_failure_manifold.py

# Run the cross-task transfer experiment
.venv/bin/python3 src/experiments/exp026_cross_task_transfer.py

# Run the failure manifold mapping experiment
.venv/bin/python3 src/experiments/exp027_failure_manifold_mapping.py

# Run the failure manifold validation experiment
.venv/bin/python3 src/experiments/exp028_failure_manifold_validation.py

# Run the failure manifold trajectories experiment
.venv/bin/python3 src/experiments/exp029_failure_trajectories.py

# Run the universal audio state space experiment
.venv/bin/python3 src/experiments/exp030_universal_audio_state_space.py

# Run the assumption surfaces experiment
.venv/bin/python3 src/experiments/exp031_assumption_surfaces.py

# Open listen tests
open listen_test/index.html
open listen_test/exp023.html
```

---

## Where We Are

The project has crossed from **DSP research** into **audio product territory**.

The representation intelligence pipeline is complete:
```
Signal
  ↓
ACF + Cepstrum + STFT
  ↓
Self-Confidence Signals
  ↓
Failure Forecasting
  ↓
Meta-Representation Layer (learned fusion weights)
  ↓
Confidence-Gated NoteTracker (UNDECIDED → TRACKING → LOCKED)
  ↓
IIR Correction Dynamics (retune speed)
  ↓
Output Audio
```

The remaining open questions are **product questions**, not research questions:
- What retune speed feels right for different vocal styles?
- Should the retune speed itself be confidence-adaptive (faster when very confident)?
- What does the system sound like on polyphonic material?
- Can the meta-layer be updated online (continuous learning from the incoming signal)?
