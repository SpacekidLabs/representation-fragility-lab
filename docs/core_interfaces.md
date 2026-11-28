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
- `PairwiseComparator`: all the aggregation (pair score, overall, divergence, outlier, consensus). The consensus needs a **quorum**: it is fused from the largest group of trusted readings that all agree pairwise, and only if at least `min_agreeing` (default 2) of them do; `report.consensus_from` names them. `min_agreeing=None` restores the old rule (fuse every trusted reading). A medium only overrides
  - `estimate_distance(a, b) -> (distance 0..1, relation label, details)`: how far apart two of its estimates are, and how they relate.
  - optionally `fuse(readings)`: how to combine trusted estimates (default: take the most confident).
  - optionally `agrees(pair)`: what counts as agreement for the quorum (default: relation `"agree"`).
- Family-aware choice of group (issue #21, Exp 047), opt-in: `PairwiseComparator(family=...)` picks the best-scoring maximal agreeing group (of at least `min_agreeing`) instead of the largest. Two scorers, both learned from labelled readings and kept as plain tables:
  - `FamilyVotes.fit(confidences, failures)`: per-lens P(right | confidence bin) and pairwise failure correlation; a group scores `sum_i p_i / (1 + sum_j R_ij)`.
  - `AgreementTable.fit((group, right) examples)`: P(right | exactly this set of lenses agrees). Better on images; `min_score=0.5` abstains when no group is likely right.
- Lead lens (issue #31, Exp 056), opt-in: `PairwiseComparator(lead="crepe")` lets one lens answer alone whenever it is trusted; the other lenses only warn, through `report.warning` (the lead's divergence, 0 = they agree). When the lead is blind or missing, `lead_fallback="abstain"` (default) gives no consensus and `"quorum"` uses the usual group choice. Use it where one lens is known to be better (CREPE for pitch, the peak counter for ECG, Radon on photos); without `lead` nothing changes.
- Probing (issue #30, Exp 057), opt-in: a medium may implement `probes(signal, candidate, context)`, returning `Probe(name, signal, predicted)`. Each is a perturbation with a known effect, plus the estimate the lenses should read on it if the candidate is right. `Analyzer(prober=Prober())` then settles every disagreement by asking the signal. The candidates are the lenses' distinct answers, trusted or not. Every lens reads every probe, and the candidate whose predictions the lenses follow most often wins. `Analysis.answer` is that choice, or the consensus where the lenses agree. `ProbeResult.score` (the winner's share of follows) is low on input with nothing to find; `Prober(min_score=...)` abstains below a threshold. Audio probes with a low-pass at 1.5× the candidate; images with blur and a known rotation. Images also have opt-in oriented blurs and a centre window (Exp 060). None beats the default, and blurring along the candidate creates what it tests, so it must not be used.
- Admission test (issue #32, Exp 058): `admission_test(frames, truths, pool, candidate, comparator, is_wrong, nulls=, groups=)` says from cached readings whether a new lens is worth adding. It reports the candidate's coverage (right where fewer than a quorum of pool lenses are), how often it is right on the frames the pool leaves unanswered, and the change in gross_all, abstention and false answers when it is added, with a group-bootstrap CI. A lens is admitted when the gross_all CI lies below 0 and false answers rise by at most 2 pts.
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
their blind spots differ, so the default stays at three. A `FamilyVotes` that knows
STFT, CQT and wavelet fail together restores the right answer with all five
(`tests/test_audio_core.py`, issue #21).

## Second medium: images (issue #6)

Images are implemented in [`src/image_medium`](../src/image_medium) with no change
to the core. The comparable quantity is the dominant orientation of a patch,
in degrees mod 180.

| Interface | Image implementation |
|---|---|
| `Medium` | `ImagePatch`: a 2-D array (H×W×C is averaged to gray), at least 16×16, values ~[0, 1]. `is_empty` = standard deviation below 1e-3 (a third of an 8-bit grey level). |
| `Representation` | Five orientation lenses, each returning `Reading(estimate=degrees or None, confidence)` with its assumptions:<br>• `TensorOrientation`: structure tensor; *assumes one dominant orientation; strong foreign edges pull the mean*.<br>• `HogOrientation`: mode of the gradient-orientation histogram; *assumes most gradient pixels share the orientation; per-pixel gradients not swamped by noise*.<br>• `FftOrientation`: strongest 2-D power-spectrum peak; *assumes a periodic texture with a peak well away from DC*.<br>• `GaborOrientation`: oriented Gabor bank, wavelengths 4–16 px, 16 orientations; *assumes structure at those scales*.<br>• `RadonOrientation`: variance of line sums per angle inside the inscribed disk; *assumes long straight structures not broken by occluders*. |
| `BlindSpotDetector` | `ConfidenceThresholdDetector(0.5)`, as for audio. |
| `Comparator` | `OrientationComparator(PairwiseComparator)`: `estimate_distance` = angular difference / 90°, relation `"agree"` (< 10°) / `"orthogonal"` (> 80°) / `"off"`; `fuse` = confidence-weighted circular median (the input that minimises the weighted angular distance to all others). |
| `PatternModel` | Not in the library yet; Exp 044 uses the Exp 043 approach on 5 confidences + 10 signed pair differences. |

Confidences are calibrated like the audio CQT and wavelet lenses: the strongest
raw contrast over 60 white-noise patches maps to 0 and the weakest 5% of clean
test patterns to 1 (`exp044_image_disagreement.py --calibrate`).

```python
from representation_image import make_image_analyzer

report = make_image_analyzer().analyze(patch).report
report.consensus, report.outlier, report.pair("fft", "radon").relation
```

**What the second medium showed about the core.** The loop, the detector and the
aggregation carried over unchanged. Two limits of the generic logic showed up:

- *A single trusted reading was enough for a consensus.* On smooth isotropic noise
  the FFT lens is confidently wrong 96% of the time while the other four are
  unconfident; the consensus followed the FFT lens (and on real unvoiced audio,
  ACF or Cepstrum). Fixed by the quorum (issue #20, Exp 046): false answers fall
  to 0–7%, at the cost of more abstentions under heavy noise.
- *Correlated lenses outvote the right one.* As with the spectral trio in audio,
  lenses fail in families (local gradient lenses vs global spectral/projection
  lenses), so the weighted median picks the family that is wrong. Exp 044's best
  single lens (Radon, 4.8% gross errors) beats the consensus (9.8%) and a learned
  choice of lens (5.6%). Exp 047 adds family-aware group choice (`AgreementTable`:
  6.8%), but even the best possible choice among trusted groups (5.6%) cannot beat
  Radon: the right lens is usually below the confidence threshold.

## Design choices

- **One comparable quantity per medium task.** Representations only become comparable once reduced to the same estimate (f0, orientation…). The raw transforms stay in `Reading.details`.
- **Assumptions are data.** `Representation.assumptions` lists the conditions under which a lens is valid, as plain text, so blind spots stay explainable.
- **Empty is not disagreement.** `Medium.is_empty` signals are reported (`Analysis.empty`), not compared.
- **No consensus without agreement.** The consensus needs at least two trusted readings that agree (Exp 046); one confident lens is not enough. If every reading is blind, `consensus` and `outlier` are `None`: when all lenses are blind, none is singled out (Exp 042/043).
- **Inspectable patterns.** `PatternModel.vectorize` must return named, interpretable numbers, not opaque embeddings.
- **Standard library only.** Numeric work belongs in a medium's implementation, so the core stays light and importable anywhere.
