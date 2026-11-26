# Core Interfaces — Design Note

The core (`src/core`, installed as `representation_core`) holds the comparison and
pattern machinery of [the vision](VISION.md) without knowing anything about a
medium. It uses only the Python standard library, and a test checks that
importing it loads no audio or numeric libraries.

```
Signal ─► Medium.validate ─► Representation.read × n ─► BlindSpotDetector.assess
                                                              │
                     PatternModel.assign ◄─ Comparator.compare ◄┘
```

`Analyzer` runs this loop for one signal and returns an `Analysis`
(`report: ComparisonReport`, optional `pattern: Pattern`).

## The five building blocks

| Interface | Responsibility | Must implement |
|---|---|---|
| `Medium` | What a signal is: validation, required context, "nothing to analyse" | `validate(signal, context)`; optional `is_empty` |
| `Representation` | One lens: a transform plus its **stated assumptions**, reduced to a comparable `Reading(estimate, confidence)` | `name`, `medium`, `assumptions`, `read(signal, context)` |
| `BlindSpotDetector` | Per reading: is this lens likely blind here? → `Assessment(safe, blind, reasons)` | `assess(reading, signal, context)` |
| `Comparator` | Across readings of one signal: pairwise agreement, overall disagreement, per-lens divergence, outlier, consensus → `ComparisonReport` | `compare(readings, assessments)` |
| `PatternModel` | Across many signals: turn reports into inspectable vectors, learn named patterns → `Pattern` | `vectorize(report)`, `fit(reports)`, `assign(report)` |

## What a new medium has to write

Most of the work is generic and already in `core.generic`:

- `ConfidenceThresholdDetector`: blind when confidence < threshold (default 0.5) or there is no estimate.
- `PairwiseComparator`: all the aggregation (pair score, overall, divergence, outlier, consensus over trusted readings). A medium only overrides
  - `estimate_distance(a, b) -> (distance 0..1, relation label, details)`: how far apart two of its estimates are, and how they relate.
  - optionally `fuse(readings)`: how to combine trusted estimates (default: take the most confident).
- `Analyzer`: the loop. It also checks that every representation belongs to the medium and that names are unique.

So a medium is: one `Medium`, n `Representation`s, one `estimate_distance`, and optionally a `PatternModel`.

## Worked example: audio pitch

Audio is implemented on the core in [`src/framework/audio.py`](../src/framework/audio.py) (issue #5).
The pitch readers live in `src/framework/disagreement.py`. Its original API
(`analyze_disagreement`, `compare_readings`, `DisagreementReport`) is kept as
an adapter over the core and gives identical results: Exp 039, 041, 042 and 043
were re-run before and after the port, with no differences.

| Interface | Audio implementation |
|---|---|
| `Medium` | `AudioFrame`: a 1-D float array; `context` must hold `sr`, and optionally `fmin`/`fmax`. `is_empty` = RMS below 1e-8 (silence). |
| `Representation` | Five pitch lenses, each returning `Reading(estimate=f0_hz or None, confidence)` and listing its assumptions:<br>• `StftPitch`: harmonic summation on the Hann-windowed spectrum; *assumes a stationary frame and resolved harmonics*.<br>• `AcfPitch`: first strong normalised-autocorrelation peak; *assumes the waveform repeats within the frame*.<br>• `CepstrumPitch`: real-cepstrum peak; *assumes many evenly spaced harmonics*.<br>• `CqtPitch`: harmonic summation on a 24-bins/octave constant-Q transform, using only partials at least 2 bins apart (up to the 11th); *assumes resolved low harmonics*.<br>• `WaveletPitch`: harmonic summation on a 6-cycle Morlet scalogram, first 3 partials only; *assumes a strong fundamental and low partials*. |
| `BlindSpotDetector` | `ConfidenceThresholdDetector(0.5)`: the "voiced" rule. Exp 042 shows a learned detector on disagreement features does better (AUC 0.95–0.99), so a `LearnedDetector` is a natural next implementation. |
| `Comparator` | `PitchComparator(PairwiseComparator)`: `estimate_distance` = `min(cents / 1200, 1)`, relation `"agree"` / `"octave above"` / `"octave below"` / `"off"`; `fuse` = confidence-weighted median f0. |
| `PatternModel` | The Exp 043 model: `vectorize` = confidences + signed pair ratios in octaves; `fit` = k-means with silhouette-chosen k; `assign` = rule-based names such as "CEP an octave low, others agree". (Still in the experiment script; moving it into the library is future work.) |

```python
from representation_intelligence import make_audio_analyzer, ALL_LENSES

analyzer = make_audio_analyzer()               # STFT + ACF + Cepstrum (default)
analysis = analyzer.analyze(frame, {"sr": 22050})
analysis.report.outlier, analysis.report.consensus, analysis.report.pair("acf", "cepstrum").relation

make_audio_analyzer(ALL_LENSES)                # all five lenses
```

**CQT and wavelet calibration.** On 2048-sample frames at 22.05 kHz, both new
lenses have a median pitch error under 1 cent on clean harmonic tones
(110–660 Hz, 1–20 harmonics). Their confidence maps the peak contrast of the
harmonic-summation curve to 0..1, with the loudest of 60 white-noise frames at
0 and the weakest 5% of clean tones at 1. CQT filters are kept at full length:
shortening them to fit the frame skews the peaks and biased the pitch by about
−11 cents.

**More lenses are not more independent votes.** When the fundamental is
missing, STFT, CQT and wavelet (all spectral, all leaning on the low partials)
jump an octave up together. With the default three lenses, ACF + Cepstrum
outvote STFT and the consensus is right. With all five, the spectral trio wins
and the consensus is wrong (`tests/test_audio_core.py`). This is Exp 043's
co-failure finding seen from the other side. Adding lenses only helps when
their blind spots differ, so the default stays at three until a fusion rule
accounts for shared blind spots.

## Sketch: images (issue #6)

| Interface | Image implementation |
|---|---|
| `Medium` | `ImagePatch`: a 2-D (or H×W×C) array; `is_empty` = near-zero variance. |
| `Representation` | Lenses estimating the same quantity, e.g. dominant orientation from gradients (*assumes sharp, noise-free edges*), from the 2-D FFT (*assumes a periodic texture*) and from a wavelet decomposition (*assumes multiscale structure*). |
| `Comparator` | `estimate_distance` = angular difference / 90°, relation `"agree"` / `"orthogonal"` / `"off"`. |
| `PatternModel` | The same Exp 043 approach on image disagreement vectors. |

## Design choices

- **One comparable quantity per medium task.** Representations only become comparable once reduced to the same estimate (f0, orientation…). The raw transforms stay in `Reading.details`.
- **Assumptions are data.** `Representation.assumptions` lists the conditions under which a lens is valid, as plain text, so blind spots stay explainable.
- **Empty is not disagreement.** `Medium.is_empty` signals are reported (`Analysis.empty`), not compared.
- **No consensus without trust.** If every reading is blind, `consensus` and `outlier` are `None`: when all lenses are blind, none is singled out (Exp 042/043).
- **Inspectable patterns.** `PatternModel.vectorize` must return named, interpretable numbers, not opaque embeddings.
- **Standard library only.** Numeric work belongs in a medium's implementation, so the core stays light and importable anywhere.
